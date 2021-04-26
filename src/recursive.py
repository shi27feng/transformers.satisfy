import torch
from torch_scatter import scatter


def backtracking(xv, xc, adj_pos, adj_neg):
    xv = xv.squeeze(-1)
    xv = torch.where(xv > 0.5, 1., 0.)
    xp, xn = xv[adj_pos[1]], (1 - xv)[adj_neg[1]]
    x = torch.cat((xp, xn))
    idx = torch.cat((adj_pos[0], adj_neg[0]))
    clause_sat = scatter(x, idx, reduce="sum")
    return min(clause_sat)
