import torch
from torch_scatter import scatter


def backtracking(v, xc, adj_pos, adj_neg):
    v = torch.where(v.squeeze(-1) > 0.5, 1., 0.)
    vp, vn = v[adj_pos[1]], (1 - v)[adj_neg[1]]
    cp = scatter(vp, adj_pos[0], reduce="sum")
    cn = scatter(vn, adj_pos[0], reduce="sum")
    return
