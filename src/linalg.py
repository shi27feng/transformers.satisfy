import torch
from torch import Tensor
from torch_scatter import scatter_add
from torch_sparse import spmm, transpose
from torch_geometric.utils.num_nodes import maybe_num_nodes
from einops import rearrange, repeat


def _spmm(indices, nz, m, n, d):
    """Sparse matrix multiplication, it supports tensor
    with dimension size more than 2, and the code is inspired by:
    "PyTorch Sparse"[https://tinyurl.com/ycn2nkdr]
    :argument
        indices (:class: `LongTensor`): tensor of indices of sparse matrix.
        nz (:class: `Tensor`): tensor of nonzero of sparse matrix.
        m (int): The first dimension of corresponding dense matrix.
        n (int): The second dimension of corresponding dense matrix.
        d (:class:`Tensor`): tensor of dense matrix
    """
    assert n == d.shape[-2]
    rows, cols = indices
    d = d if d.dim() > 1 else d.unsqueeze(-1)
    out = d[..., cols, :] * nz.unsqueeze(-1)
    return scatter_add(out, rows, dim=-2, dim_size=m)


def batched_spmm(nzt, adj, x, m=None, n=None):
    """
    Args:
        nzt: Tensor [num_edges, heads]    -- non-zero tensor
        adj: Tensor or list(Tensor)       -- adjacency matrix (COO)
        x:   Tensor [num_nodes, channels] -- feature matrix
        m:   int
        n:   int
    """
    _, heads = nzt.size()
    num_nodes, channels = x.size()
    # preparation of data
    x_ = repeat(x, '... n c -> ... (h n) c', h=heads)
    nzt_ = rearrange(nzt, '... e h -> ... (h e)')
    if isinstance(adj, Tensor):
        m = maybe_num_nodes(adj[0], m)
        n = max(num_nodes, maybe_num_nodes(adj[1], n))
        offset = torch.tensor([[m], [n]]).to(x_.device)
        adj_ = torch.cat([adj + offset * i for i in range(heads)], dim=1)
    else:  # adj is list of adjacency matrices
        assert heads == len(
            adj), "the number of heads and the number of adjacency matrices are not matched"
        m = max([maybe_num_nodes(adj_[0], m) for adj_ in adj])
        n = max([maybe_num_nodes(adj_[1], n) for adj_ in adj])
        offset = torch.tensor([[m], [n]])
        adj_ = torch.cat([adj[i] + offset * i for i in range(heads)], dim=1)

    return spmm(adj_, nzt_, heads * m, heads * n, x_)


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
    else:  # adj is a list of Tensor
        adj_ = [None] * value.shape[1]
        vs = torch.zeros(value.shape)
        m = max([maybe_num_nodes(a_[0], m) for a_ in adj])
        n = max([maybe_num_nodes(a_[1], n) for a_ in adj])
        for j in range(len(adj)):
            adj_[j], vs[:, j] = transpose(adj[j], value[:, j], m, n)
        return adj_, vs
