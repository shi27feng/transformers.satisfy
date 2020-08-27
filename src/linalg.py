
import torch
from torch import Tensor
from torch_sparse import spmm


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
    if isinstance(adj, Tensor):
        return
    else:
        return
    return
