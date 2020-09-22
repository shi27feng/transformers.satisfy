import copy
from abc import ABC
from inspect import Parameter as Pr
from typing import Union, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as fn
from torch import Tensor
from torch.nn import Parameter, Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import softmax

from linalg import batched_spmm, batched_transpose
from utils import self_loop_augment


def clones(module, k):
    return nn.ModuleList(
        copy.deepcopy(module) for i in range(k)
    )


class LayerNorm(nn.Module, ABC):
    """Construct a layer-norm module (See citation for details)."""

    def __init__(self, in_channels, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.alpha = nn.Parameter(torch.ones(in_channels))
        self.beta = nn.Parameter(torch.zeros(in_channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.beta


class SublayerConnection(nn.Module, ABC):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity we apply the norm first as opposed to last.
    """

    def __init__(self, in_channels, drop_rate):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(in_channels)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer function that maintains the same in_channels."""
        if isinstance(sublayer, Tensor):
            return x + self.dropout(self.norm(sublayer))  # TODO check reverse order of norm and sublayer
        return x + self.dropout(sublayer(self.norm(x)))
        # if x is Tensor:
        #     return x + self.dropout(sublayer(self.norm(x)))
        # else:
        #     xv, xc = x
        #     xv, xc = sublayer((self.norm(xv), self.norm(xc)))
        #     return self.dropout(xv) + x[0], self.dropout(xc) + x[1]


class EncoderLayer(nn.Module, ABC):
    """Encoder is made up of two sub-layers, self-attn and feed forward (defined below)"""

    def __init__(self,
                 in_channels,
                 hidden_dims,
                 out_channels,
                 num_meta_paths,
                 self_att_heads=4,
                 cross_att_heads=4,
                 drop_rate=0.5):
        super(EncoderLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dims = hidden_dims
        # weights for meta-paths
        self.lit_path_weights = nn.Parameter(torch.ones(num_meta_paths))
        self.cls_path_weights = nn.Parameter(torch.ones(num_meta_paths))

        self.self_lit_attentions = clones(HGAConv(
            in_channels, hidden_dims, heads=self_att_heads), num_meta_paths)
        self.self_cls_attentions = clones(HGAConv(
            in_channels, hidden_dims, heads=self_att_heads), num_meta_paths)

        self.sublayer_lit = clones(SublayerConnection(in_channels, drop_rate), num_meta_paths)
        self.sublayer_cls = clones(SublayerConnection(in_channels, drop_rate), num_meta_paths)
        self.cross_attention_pos = HGAConv((hidden_dims, hidden_dims),
                                           out_channels, heads=cross_att_heads)
        self.cross_attention_neg = HGAConv((hidden_dims, hidden_dims),
                                           out_channels, heads=cross_att_heads)

    @staticmethod
    def _attention_meta_path(x, meta_paths, att_layers, sublayers, path_weights):
        assert len(att_layers) == len(meta_paths), "the length should match"
        res = torch.zeros(x.shape)
        # TODO try to use batched matrix for meta-paths
        #   e.g. concatenate adj of meta-path as one diagonalized matrix, and stack x
        for i in range(len(att_layers)):  # they are not sequential, but in reduction mode
            res += path_weights[i] * sublayers[i](x, att_layers[i](x, meta_paths[i][0]))  # TODO 
        return res

    def forward(self, xv, xc, meta_paths_lit, meta_paths_cls, adj_pos, adj_neg):
        # xv = self._attention_meta_path(xv, meta_paths_lit, self.self_lit_attentions, self.lit_path_weights)
        # xc = self._attention_meta_path(xc, meta_paths_cls, self.self_cls_attentions, self.cls_path_weights)
        xv = self._attention_meta_path(xv, meta_paths_lit, self.self_lit_attentions, self.sublayer_lit, self.lit_path_weights)
        xc = self._attention_meta_path(xc, meta_paths_cls, self.self_cls_attentions, self.sublayer_cls, self.cls_path_weights)
        '''
        xc = self.sublayer_cls(xc, lambda x: self._attention_meta_path(x,
                                                                       meta_paths_cls,
                                                                       self.self_cls_attentions,
                                                                       self.cls_path_weights))
        '''
        xv_pos, xc_pos = self.cross_attention_pos((xv, xc), adj_pos)
        xv_neg, xc_neg = self.cross_attention_neg((xv, xc), adj_neg)
        return xv_pos + xv_neg, xc_pos + xc_neg  # TODO is xv_pos + xv_neg appropriate?


class DecoderLayer(nn.Module, ABC):
    """Decoder is made of self-attn, src-attn, and feed forward (defined below)"""

    def __init__(self,
                 in_channels,
                 hidden_dims,
                 out_channels,
                 cross_att_heads=4,
                 drop_rate=0.5,
                 feed_forward=None):
        super(DecoderLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dims = hidden_dims

        self.attn_pos = HGAConv((in_channels, in_channels),
                                hidden_dims, heads=cross_att_heads)
        self.attn_neg = HGAConv((in_channels, in_channels),
                                hidden_dims, heads=cross_att_heads)

        self.ff_v = nn.Linear(hidden_dims, out_channels)
        self.ff_c = nn.Linear(hidden_dims, out_channels)

        self.sublayer = clones(SublayerConnection(out_channels, drop_rate), 2)  # 4

    def forward(self, xv, xc, adj_pos, adj_neg):
        """Follow Figure 1 (right) for connections."""
        # xv_pos, xc_pos = self.sublayer[0]((xv, xc), lambda x: self.attn_pos(x, adj_pos))
        # xv_neg, xc_neg = self.sublayer[1]((xv, xc), lambda x: self.attn_neg(x, adj_neg))
        xv_pos, xc_pos = self.attn_pos((xv, xc), adj_pos)
        xv_neg, xc_neg = self.attn_neg((xv, xc), adj_neg)
        return self.sublayer[0](xv_pos + xv_neg, self.ff_v), \
            self.sublayer[1](xc_pos + xc_neg, self.ff_c)


class HGAConv(MessagePassing):
    """
    Heterogeneous Graph Attention Convolution
    """

    def _forward_unimplemented(self, *in_tensor: Any) -> None:
        pass

    def __init__(self,
                 in_channels: Union[int, Tuple[int, int]],
                 out_channels: int,
                 heads: int = 1,
                 concat: bool = True,
                 negative_slope: float = 0.2,
                 dropout: float = 0.,
                 use_self_loops: bool = True,
                 bias: bool = True, **kwargs):
        super(HGAConv, self).__init__(aggr='add', node_dim=0, **kwargs)

        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.add_self_loops = use_self_loops

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, out_channels, bias=False)

            self.lin_r = self.lin_l
        else:
            self.lin_l = Linear(in_channels[0], out_channels, False)
            self.lin_r = Linear(in_channels[1], out_channels, False)

        self.att_l = Parameter(torch.Tensor(out_channels, heads))
        self.att_r = Parameter(torch.Tensor(out_channels, heads))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(out_channels * heads))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        glorot(self.att_l)
        glorot(self.att_r)
        zeros(self.bias)

    @staticmethod
    def edge_score(adj, a_l, a_r):
        """
        Args:
            adj: adjacency matrix [2, num_edges] or (heads, [2, num_edges])
            a_l: Tensor           [num_nodes, heads]
            a_r: Tensor           [num_nodes, heads]
        """
        if isinstance(adj, Tensor):
            return a_l[adj[0], :] + a_r[adj[1], :]  # [num_edges, heads]
        a = []
        for i in range(len(adj)):
            a[i] = a_l[adj[i][0], i] + a_r[adj[i][1], i]
        return a  # (heads, [num_edges, 1])

    def forward(self, x, adj, size=None, return_attention_weights=None):
        """
        Args:
            x: Union[Tensor, PairTensor]
            adj: Tensor[2, num_edges] or list of Tensor
            size: Size
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(adj, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        h, c = self.heads, self.out_channels
        # assert (not isinstance(adj, Tensor)) and h == len(adj), 'Number of heads is number of adjacency matrices'

        x_l, x_r, alpha_l, alpha_r = None, None, None, None

        if isinstance(x, Tensor):
            x_l, x_r = x, None
        else:
            x_l, x_r = x[0], x[1]
        assert x_l.dim() == 2, 'Static graphs not supported in `HGAConv`.'
        x_l = self.lin_l(x_l)
        alpha_l = torch.mm(x_l, self.att_l)
        if x_r is not None:
            x_r = self.lin_r(x_r)
            alpha_r = torch.mm(x_r, self.att_r)
        else:
            alpha_r = torch.mm(x_l, self.att_r)
        assert x_l is not None
        assert alpha_l is not None

        if self.add_self_loops:
            num_nodes = x_l.size(0)
            num_nodes = size[1] if size is not None else num_nodes
            num_nodes = x_r.size(0) if x_r is not None else num_nodes
            if isinstance(adj, Tensor):
                adj = self_loop_augment(num_nodes, adj)
            else:
                for i in range(len(adj)):
                    adj[i] = self_loop_augment(num_nodes, adj[i])

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        xpar = (x_l, x_r) if x_r is not None else x_l
        alphapar = (alpha_l, alpha_r)
        out = self.propagate(adj,
                             x = xpar,
                             alpha = alphapar,
                             size=size)

        alpha = self._alpha
        self._alpha = None

        if self.concat:  # TODO if 'out' is Tuple(Tensor, Tensor)
            if isinstance(out, Tensor):
                out = out.view(-1, self.heads * self.out_channels)
            else:
                out = (out[0].view(-1, self.heads * self.out_channels),
                       out[1].view(-1, self.heads * self.out_channels))
        else:
            if isinstance(out, Tensor):
                out = out.mean(dim=1)
            else:
                out = (out[0].mean(dim=1), out[1].mean(dim=1))
        if self.bias is not None:
            if isinstance(out, Tensor):
                out += self.bias
            else:
                out = (out[0] + self.bias, out[1] + self.bias)
        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            return out, (adj, alpha)
        else:
            return out

    def propagate(self, adj, size=None, **kwargs):
        # propagate_type: (x: OptPairTensor, alpha: PairTensor)
        size = self.__check_input__(adj, size)

        x = kwargs.get('x', Pr.empty)  # OptPairTensor
        alpha = kwargs.get('alpha', Pr.empty)  # PairTensor
        score = self.edge_score(adj=adj, a_l=alpha[0], a_r=alpha[1])
        out = self.message_and_aggregate(adj, x=x, score=score)

        return self.update(out)

    def _attention(self, adj, score):  # score: [num_edges, heads]
        alpha = fn.leaky_relu(score, self.negative_slope)
        alpha = softmax(alpha, adj[1])
        self._alpha = alpha
        return fn.dropout(alpha, p=self.dropout, training=self.training)

    def message_and_aggregate(self,
                              adj,
                              x,
                              score):
        """
        Args:
            adj:   Tensor or list(Tensor)
            x:     Union(Tensor, PairTensor) for bipartite graph
            score: Tensor or list(Tensor)
        """
        # for bipartite graph, x_l -> out_ and x_r -> out_l (interleaved)
        x_l, x_r, out_, out_l = None, None, None, None
        n, m = 0, 0
        if isinstance(x, Tensor):
            x_l = x
        else:
            x_l, x_r = x[0], x[1]
            (m, c2) = x_r.size()
            out_l = torch.zeros((m, c2, self.heads))

        if isinstance(adj, Tensor):
            alpha = self._attention(adj, score)  # [num_edges, heads]
        else:  # adj is list of Tensor
            alpha = []
            for i in range(self.heads):
                alpha.append(self._attention(adj[i], score[i]))

        out_ = batched_spmm(alpha, adj, x_l)
        if x_r is None:
            return out_.permute(1, 0, 2)
        else:
            adj, alpha = batched_transpose(adj, alpha)
            out_l = batched_spmm(alpha, adj, x_r)
            return out_.permute(1, 0, 2), out_l.permute(1, 0, 2)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

if __name__=="__main__":
    import models
    from args import make_args
    args = make_args()

    from data import SATDataset
    ds = SATDataset('dataset', 'RND3SAT/uf50-218', False)
    last_trn, last_val = int(len(ds)), int(len(ds))
    train_ds = ds[: last_trn]
    valid_ds = ds[last_trn: last_val]
    test_ds = ds[last_val:]

    test_data = train_ds[1]
    edge_index_pos = test_data.edge_index_pos
    edge_index_neg = test_data.edge_index_neg
    xv = torch.rand(50, 1)
    xc = torch.ones(218, 1)
    variable_count = max(max(edge_index_pos[1]), max(edge_index_neg[1]))+1
    clause_count = max(max(edge_index_pos[0]), max(edge_index_neg[0]))+1
    edge_count = len(edge_index_pos[1])

    model = models.make_model(args)
    #from torchvision import models
    #model = models.vgg16()
    print(model)

    literal_assignment = model(xv, xc, edge_index_pos, edge_index_neg)
    loss_func = loss_func = SimpleLossCompute(par_sm, par_sg, "cuda")
    loss_of_this_assignent = loss_func(x_s, edge_index_pos, edge_index_neg)
   