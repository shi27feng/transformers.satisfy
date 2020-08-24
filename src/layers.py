import torch.nn as nn
import copy
from typing import Union, Tuple, Optional, Any
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)
import torch
from torch import Tensor
import torch.nn.functional as fn
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros


def clones(module, k):
    return nn.ModuleList(
        copy.deepcopy(module) for _ in range(k)
    )


def attention(h, adj):
    return


def self_loop_augment(x_l, x_r, size, adj):
    num_nodes = x_l.size(0)
    num_nodes = size[1] if size is not None else num_nodes
    num_nodes = x_r.size(0) if x_r is not None else num_nodes
    adj, _ = remove_self_loops(adj)
    adj, _ = add_self_loops(adj, num_nodes=num_nodes)
    return adj


class HGAConv(MessagePassing):
    """
    Heterogeneous Graph Attention Convolution
    """
    def _forward_unimplemented(self, *in_tensor: Any) -> None:
        pass

    def message_and_aggregate(self, adj_t: SparseTensor) -> Tensor:
        pass

    _alpha: OptTensor

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

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
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
            x: Tensor
            adj: Tensor or list of Tensor
            size: Size
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        h, c = self.heads, self.out_channels
        assert (not isinstance(x, Tensor)) and h == len(adj), 'Number of heads is number of adjacency matrices'

        x_l = None
        x_r = None
        alpha_l = None
        alpha_r = None
        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `HGAConv`.'
            x_l = x_r = self.lin_l(x).view(-1, h, c)
            alpha_l = alpha_r = (x_l * self.att_l).sum(dim=-1)  # dot product
        else:   # for bipartite graph
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2, 'Static graphs not supported in `HGAConv`.'
            x_l = self.lin_l(x_l).view(-1, h, c)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, h, c)
                alpha_r = (x_r * self.att_r).sum(dim=-1)

        assert x_l is not None
        assert alpha_l is not None

        if self.add_self_loops:
            if isinstance(adj, Tensor):
                adj = self_loop_augment(x_l, x_r, size, adj)
            else:
                for i in range(len(adj)):
                    adj[i] = self_loop_augment(x_l, x_r, size, adj[i])

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(adj, x=(x_l, x_r),
                             alpha=(alpha_l, alpha_r), size=size)

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
            if isinstance(adj, Tensor):
                return out, (adj, alpha)
            elif isinstance(adj, SparseTensor):
                return out, adj.set_value(alpha, layout='coo')
        else:
            return out

    def message(self,
                x_j: Tensor,
                alpha_j: Tensor,
                alpha_i: OptTensor,
                index: Tensor,
                ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = fn.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = fn.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

