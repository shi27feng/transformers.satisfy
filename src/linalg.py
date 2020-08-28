
import torch
from torch import Tensor
from torch_sparse import spmm, transpose
from torch_geometric.utils.num_nodes import maybe_num_nodes


def batched_spmm(nzt, adj, x, m=None, n=None):
    """
    Args:
        nzt: Tensor [num_edges, heads]    -- non-zero tensor
        adj: Tensor or list(Tensor)       -- adjacency matrix (COO)
        x:   Tensor [num_nodes, channels] -- feature matrix
        m:   int
        n:   int
    """
    heads, num_edges = nzt.size()
    num_nodes, channels = x.size()
    # preparation of data
    out = None
    if isinstance(adj, Tensor):
        x_ = torch.cat(heads * [x])  # duplicate x for heads times
        m = maybe_num_nodes(adj[0], m)
        n = maybe_num_nodes(adj[1], max(num_nodes, n))
        offset = torch.tensor([[m], [n]])
        adj_ = torch.cat([adj + offset * i for i in range(heads)], dim=1)
        nzt_ = nzt.view(1, -1)
        out = spmm(adj_, nzt_, m * heads, n * heads, x_)
    else:  # adj is list of adjacency matrices
        m = max([maybe_num_nodes(adj_[0], m) for adj_ in adj])
        out = torch.zeros([m * heads, channels])
        for i in range(heads):
            out[m * i: m * (i + 1), :] = spmm(adj[i], nzt[:, i], m, num_nodes, x)
    return out.view(-1, m, channels)    # [heads, m, channels]


def batched_transpose(adj, value, m=None, n=None):
    """
    Args:
        adj: Tensor or list of Tensor
        value: Tensor [num_edges, ]
        m: int
        n: int
    """
    if isinstance(adj, Tensor):
        m = maybe_num_nodes(adj[0], m)
        n = maybe_num_nodes(adj[1], n)
        return transpose(adj, value, m, n)
    else:   # adj is a list of Tensor
        adj_ = [None] * value.shape[1]
        vs = torch.zeros(value.shape)
        m = max([maybe_num_nodes(a_[0], m) for a_ in adj])
        n = max([maybe_num_nodes(a_[1], n) for a_ in adj])
        for j in range(len(adj)):
            adj_[j], vs[:, j] = transpose(adj[j], value[:, j], m, n)
        return adj_, vs
