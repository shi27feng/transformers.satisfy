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
        copy.deepcopy(module) for _ in range(k)
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


class EncoderLayer(nn.Module, ABC):
    """Encoder is made up of two sub-layers, self-attn and feed forward (defined below)"""
    def __init__(self, size, feed_forward, dropout, self_attn=None):
        super(EncoderLayer, self).__init__()

        self.literals_weights = nn.Parameter(torch.ones(4))  # for 4 types
        self.clauses_weights = nn.Parameter(torch.ones(4))  # for 4 types

        self.self_attn_lit = self_attn
        self.self_attn_cls = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x):
        """Follow Figure 1 (left) for connections."""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module, ABC):
    """Decoder is made up of three sub-layers:
        self-attn, src-attn, and feed forward (defined below)"""

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        """
        """
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        """Follow Figure 1 (right) for connections."""
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class SublayerConnection(nn.Module, ABC):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity we apply the norm first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer function that maintains the same size."""
        return x + self.dropout(sublayer(self.norm(x)))


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
            self.lin_l = Linear(in_channels, heads * out_channels, bias=False)
            self.lin_r = self.lin_l
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels, False)
            self.lin_r = Linear(in_channels[1], heads * out_channels, False)

        self.att_l = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
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
            a_l: Tensor           [N, heads]
            a_r: Tensor           [N, heads]
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
        assert (not isinstance(x, Tensor)) and h == len(adj), 'Number of heads is number of adjacency matrices'

        x_l, x_r, alpha_l, alpha_r = None, None, None, None

        if isinstance(x, Tensor):
            x_l, x_r = x, None
        else:
            x_l, x_r = x[0], x[1]
        assert x_l.dim() == 2, 'Static graphs not supported in `HGAConv`.'
        x_l = self.lin_l(x_l).view(-1, h, c)  # dims: (N, h, c)
        alpha_l = (x_l * self.att_l).sum(dim=-1)
        if x_r is not None:
            x_r = self.lin_r(x_r).view(-1, h, c)  # dims: (N, h, c)
            alpha_r = (x_r * self.att_r).sum(dim=-1)  # reduce((N, h, c) x (h, c), c) = (N, h)
        else:
            alpha_r = (x_l * self.att_r).sum(dim=-1)

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
        out = self.propagate(adj,
                             x=(x_l, x_r),
                             alpha=(alpha_l, alpha_r),
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
