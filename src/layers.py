import copy
from abc import ABC
from inspect import Parameter as Pr
from typing import Union, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as fn
from torch import Tensor
from torch.nn import Parameter, Linear
from torch.nn.functional import relu
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import softmax

from linalg import spmm_, transpose_
from utils import self_loop_augment


def clones(module, k):
    return nn.ModuleList(
        copy.deepcopy(module) for i in range(k)
    )


class SublayerConnection(nn.Module, ABC):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity we apply the norm first as opposed to last.
    """

    def __init__(self, in_channels, drop_rate):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(in_channels)
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
        self.var_path_weights = nn.Parameter(torch.ones(num_meta_paths))
        self.cls_path_weights = nn.Parameter(torch.ones(num_meta_paths))

        self.self_var_attentions = clones(HGAConv(
            hidden_dims, hidden_dims, heads=self_att_heads, use_self_loops=True), num_meta_paths)
        self.self_cls_attentions = clones(HGAConv(
            hidden_dims, hidden_dims, heads=self_att_heads, use_self_loops=True), num_meta_paths)

        # self.sublayer_var = clones(SublayerConnection(hidden_dims, drop_rate), num_meta_paths)
        # self.sublayer_cls = clones(SublayerConnection(hidden_dims, drop_rate), num_meta_paths)
        self.sublayer_var = SublayerConnection(hidden_dims, drop_rate)
        self.sublayer_cls = SublayerConnection(hidden_dims, drop_rate)
        self.cross_attention_pos = HGAConv((hidden_dims, hidden_dims),
                                           out_channels, heads=cross_att_heads)
        self.cross_attention_neg = HGAConv((hidden_dims, hidden_dims),
                                           out_channels, heads=cross_att_heads)
        self.var_embedding = Linear(in_channels, hidden_dims, False)
        self.cls_embedding = Linear(in_channels, hidden_dims, False)

    '''
    @staticmethod
    def _attention_meta_path(x, meta_paths, att_layers, sublayers, path_weights):
        assert len(att_layers) == len(meta_paths), "the length should match"
        res = torch.zeros(x.shape).to(x.device)
        # TODO try to use batched matrix for meta-paths
        #   e.g. concatenate adj of meta-path as one diagonalized matrix, and stack x
        for i in range(len(att_layers)):  # they are not sequential, but in reduction mode
            res += path_weights[i] * sublayers[i](x, att_layers[i](x, meta_paths[i]))  # TODO 
        return res
    '''

    @staticmethod
    def _attention_meta_path(x, meta_paths, att_layers, path_weights):
        assert len(att_layers) == len(meta_paths), "the length should match"
        res = torch.zeros(x.shape).to(x.device)
        # TODO try to use batched matrix for meta-paths
        #   e.g. concatenate adj of meta-path as one diagonalized matrix, and stack x
        for i in range(len(att_layers)):  # they are not sequential, but in reduction mode
            res += path_weights[i] * relu(att_layers[i](x, meta_paths[i]))  # TODO 
        return res

    def forward(self, xv, xc, meta_paths_var, meta_paths_cls, adj_pos, adj_neg):
        xv = fn.relu(self.var_embedding(xv))
        xc = fn.relu(self.cls_embedding(xc))

        xv = relu(self.sublayer_var(xv, (lambda x: self._attention_meta_path(x,
                                                                             meta_paths_var,
                                                                             self.self_var_attentions,
                                                                             self.var_path_weights))))
        xc = relu(self.sublayer_cls(xc, (lambda x: self._attention_meta_path(x,
                                                                             meta_paths_cls,
                                                                             self.self_cls_attentions,
                                                                             self.cls_path_weights))))
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
        xv_pos, xc_pos = self.attn_pos((xv, xc), adj_pos)
        xv_neg, xc_neg = self.attn_neg((xv, xc), adj_neg)
        return self.sublayer[0](xv_pos + xv_neg, lambda x: fn.relu(self.ff_v(x))), \
            self.sublayer[1](xc_pos + xc_neg, lambda x: fn.relu(self.ff_c(x)))


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
                 concat: bool = False,
                 negative_slope: float = 0.2,
                 dropout: float = 0.,
                 use_self_loops: bool = False,  # Set to False for debug
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

        self.att_l = Parameter(torch.ones(out_channels, heads, dtype=torch.float))
        self.att_r = Parameter(torch.ones(out_channels, heads, dtype=torch.float))

        if bias and concat:
            self.bias = Parameter(torch.ones(out_channels * heads, dtype=torch.float))
        elif bias and not concat:
            self.bias = Parameter(torch.ones(out_channels, dtype=torch.float))
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
        if isinstance(adj, Tensor):  # [num_edges, heads]
            return a_l[adj[1], :] + a_r[adj[0], :] 
        a = []
        for i in range(len(adj)):
            a[i] = a_l[adj[i][1], i] + a_r[adj[i][0], i]
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

        x_l, x_r, alpha_l, alpha_r, alpha_l_, alpha_r_ = None, None, None, None, None, None

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
            alpha_r_ = torch.mm(x_l, self.att_r)
            alpha_l_ = torch.mm(x_r, self.att_l)
            self.add_self_loops = False
        else:
            alpha_r = torch.mm(x_l, self.att_r)
        assert x_l is not None
        assert alpha_l is not None

        if self.add_self_loops:
            num_nodes = x_l.size(0)
            num_nodes = size[1] if size is not None else num_nodes
            num_nodes = x_r.size(0) if x_r is not None else num_nodes
            if isinstance(adj, Tensor):
                adj = self_loop_augment(num_nodes, adj)  # TODO Bug found
            else:
                for i in range(len(adj)):
                    adj[i] = self_loop_augment(num_nodes, adj[i])

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        x_ = (x_l, x_r) if x_r is not None else x_l
        alpha = (alpha_l, alpha_r)
        alpha_ = (alpha_l_, alpha_r_)
        out = self.propagate(adj,
                             x=x_,
                             alpha=alpha,
                             alpha_=alpha_,
                             size=size)

        alpha = self._alpha
        self._alpha = None

        if self.concat:  # TODO if 'out' is Tuple(Tensor, Tensor)
            if isinstance(out, Tensor):
                out = out.reshape(-1, self.heads * self.out_channels)
            else:
                out = (out[0].reshape(-1, self.heads * self.out_channels),
                       out[1].reshape(-1, self.heads * self.out_channels))
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
        if not isinstance(x, Tensor):
            alpha_ = kwargs.get('alpha_', Pr.empty)
            score_ = self.edge_score(adj=adj, a_l=alpha_[1], a_r=alpha_[0])
            score = (score, score_)
        out = self.message_and_aggregate(adj, x=x, score=score)
        return self.update(out)

    def _attention(self, adj, score):  # score: [num_edges, heads]
        alpha = fn.leaky_relu(score, self.negative_slope)
        alpha = softmax(alpha, adj[1])
        self._alpha = alpha
        return fn.dropout(alpha, p=self.dropout, training=self.training)

    def message_and_aggregate(self, adj, x, score):
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
            n = m = x_l.shape[0]
        else:
            x_l, x_r = x[0], x[1]
            (m, c2) = x_r.size()
            n = x_l.size(0)
            out_l = torch.zeros((m, c2, self.heads))

        alpha_ = None
        if isinstance(adj, Tensor):
            if isinstance(score, Tensor):
                alpha = self._attention(adj, score)  # [num_edges, heads]
            else:
                alpha = self._attention(adj, score[0])  # [num_edges, heads]
                alpha_ = self._attention(torch.stack((adj[1], adj[0])), score[1])  # [num_edges, heads]

        else:  # adj is list of Tensor
            alpha = []
            for i in range(self.heads):
                alpha.append(self._attention(adj[i], score[i]))

        out_ = spmm_(adj, alpha, m, n, x_l, dim=-2)  # [m, (h c)], n reduced
        if x_r is None:
            return out_  # [num_nodes, heads, channels]
            # return out_.permute(1, 0, 2)  # [heads, num_nodes, channels]
        else:
            adj, alpha_ = transpose_(adj, alpha_)
            out_l = spmm_(adj, alpha_, n, m, x_r, dim=-2)  # [n, (h c)], m reduced
            return out_l, out_
            # return out_l.permute(1, 0, 2), out_.permute(1, 0, 2)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


if __name__ == "__main__":
    import models
    from args import make_args
    from torch_geometric.data import DataLoader
    from loss import LossCompute

    args = make_args()

    from data import SatDataset

    ds = SatDataset('dataset', 'RND3SAT/uf50-218', False)
    last_trn, last_val = int(len(ds)), int(len(ds))
    train_ds = ds[: last_trn]
    valid_ds = ds[last_trn: last_val]
    test_ds = ds[last_val:]

    model = models.make_model(args)
    from optimizer import get_std_opt

    opt = get_std_opt(model, args)
    loader = DataLoader(ds[0:4 * 64], batch_size=64)
    loss_func = LossCompute(30, 100, opt)

    for test_data in enumerate(loader):
        model.encoder.reset()
        edges_pos = test_data.edges_pos
        edges_neg = test_data.edges_neg
        xv = test_data.xv
        xc = test_data.xc
        literal_assignment = model(xv, xc, edges_pos, edges_neg)

        print(literal_assignment.shape)
        loss_of_this_assignment = loss_func(literal_assignment, edges_pos, edges_neg)
        print(loss_of_this_assignment)
        print()
        '''
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(name, param.grad.sum())
            else:
                print(name, param.grad)
        '''
        # print(f"loss_of_this_assignent: {loss_of_this_assignent}")
        # print("End of program")
