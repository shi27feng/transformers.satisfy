import copy
from typing import Union, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as fn
from torch import Tensor
from torch.nn import Parameter, Linear
from inspect import Parameter as Pr
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (OptTensor, PairTensor)
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_sparse import SparseTensor
from linalg import batched_spmm


def clones(module, k):
    return nn.ModuleList(
        copy.deepcopy(module) for _ in range(k)
    )


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


def self_loop_augment(num_nodes, adj):
    adj, _ = remove_self_loops(adj)
    adj, _ = add_self_loops(adj, num_nodes=num_nodes)
    return adj


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

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

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
        score = edge_score(adj=adj, a_l=alpha[0], a_r=alpha[1])
        out = self.message_and_aggregate(adj, x=x, score=score)

        return self.update(out)

    def _attention(self, adj, score):  # score: [num_edges, heads]
        alpha = fn.leaky_relu(score, self.negative_slope)
        alpha = softmax(alpha, adj[1])
        self._alpha = alpha
        return fn.dropout(alpha, p=self.dropout, training=self.training)

    def message_and_aggregate(self,
                              adj,     # Tensor or list(Tensor)
                              x,       # Union(Tensor, PairTensor) for bipartite graph
                              score):  # Tensor or list(Tensor)
        n, c = x.size()
        x_l, x_r, out_l, out_r = None, None, None, None
        if isinstance(x, Tensor):
            x_l = x
            n, c = x.size()
            out_l = torch.zeros((n, c, self.heads))
        else:
            x_l, x_r = x[0], x[1]
            (n, c1), (m, c2) = x_l.size(), x_r.size()
            out_l = torch.zeros((n, c1, self.heads))
            out_r = torch.zeros((m, c2, self.heads))

        if isinstance(adj, Tensor):
            alpha = self._attention(adj, score)  # [num_edges, heads]
            # sparse matrix multiplication of X and A (attention matrix)
        else:
            alpha = []
            for i in range(self.heads):
                alpha.append(self._attention(adj[i], score[i]))
        out = batched_spmm(alpha, adj, x_l)

        return out.permute(1, 0, 2)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
