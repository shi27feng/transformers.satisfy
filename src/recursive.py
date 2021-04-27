import torch
from torch_scatter import scatter
from linalg import transpose_


def potential_clauses(v, xc, adj_pos, adj_neg):
    v = torch.where(v.squeeze(-1) > 0.5, 1, 0)
    vp, vn = v[adj_pos[1]], (1 - v)[adj_neg[1]]
    scp = scatter(vp, adj_pos[0], reduce="sum")  # satisfied positive clauses
    scn = scatter(vn, adj_pos[0], reduce="sum")  # satisfied negative clauses
    adj_pos_t, _ = transpose_(adj_pos)
    adj_neg_t, _ = transpose_(adj_neg)
    return


def test():
    return


if __name__ == "__main__":
    test()
