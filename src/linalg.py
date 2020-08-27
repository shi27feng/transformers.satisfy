
import torch
from torch import Tensor
from torch_sparse import spmm
from torch_geometric.utils.num_nodes import maybe_num_nodes


def batched_spmm(nzt, adj, x, n=None, m=None):
    """
    Args:
        nzt: Tensor [num_edges, heads]    -- non-zero tensor
        adj: Tensor or list(Tensor)       -- adjacency matrix (COO)
        x:   Tensor [num_nodes, channels] -- feature matrix
        n:   int, S[m, n]
        m:   int, D[n, c]
    """
    heads, num_edges = nzt.size()
    num_nodes, channels = x.size()
    # preparation of data
    out = None
    if isinstance(adj, Tensor):
        x_ = torch.cat(heads * [x])  # duplicate x for heads times
        n = maybe_num_nodes(adj[1], max(num_nodes, n))
        m = maybe_num_nodes(adj[0], m)
        offset = torch.tensor([[m], [n]])
        adj_ = torch.cat([adj + offset * i for i in range(heads)], dim=1)
        nzt_ = nzt.view(1, -1)
        out = spmm(adj_, nzt_, n * heads, channels, x_)
    else:  # adj is list of adjacency matrices
        m = max([maybe_num_nodes(adj_[0], m) for adj_ in adj])
        out = torch.zeros([m * heads, channels])
        for i in range(heads):
            out[m * i: m * (i + 1), :] = spmm(adj[i], nzt[:, i], num_nodes, channels, x)
    return out.view(-1, m, channels)    # [heads, m, channels]
