import torch
from torch_scatter import scatter_sum

from linalg import transpose_


def partial_adj(adj, nodes):
    return torch.cat([adj[(adj[0] == nodes[i]).nonzero(as_tuple=True)[0]]
                      for i in range(nodes.shape[-1])], dim=0)


def potential_clauses(v, xc, adj_pos, adj_neg):
    adj_pos_t, adj_neg_t = transpose_(adj_pos)[0], transpose_(adj_neg)[0]

    v = torch.where(v.squeeze(-1) > 0.5, 1, 0)

    # get unsatisfied positive/negative clauses
    vp, vn = v[adj_pos[1]], (1 - v)[adj_neg[1]]
    ucp = torch.where(scatter_sum(vp, adj_pos[0]) > 0, 0, 1)
    ucn = torch.where(scatter_sum(vn, adj_pos[0]) > 0, 0, 1)

    # collect variables of unsatisfied clauses
    vp, vn = ucp[adj_pos_t[1]], ucn[adj_neg_t[1]]
    uvp = torch.where(scatter_sum(vp, adj_pos_t[0]) > 0, 1, 0)
    uvn = torch.where(scatter_sum(vn, adj_pos_t[0]) > 0, 1, 0)
    v_ = torch.where((uvp + uvn) > 0, 1, 0)
    if (1 - v_).sum() == 0:
        return None

    # determine the clauses to remove
    #: v_i == V(i) and v_i \in (1 - v_)
    vp, vn = ((1 - v_) * v)[adj_pos[1]], ((1 - v_) * (1 - v))[adj_neg[1]]
    ucp = torch.where(scatter_sum(vp, adj_pos[0]) > 0, 0, 1)
    ucn = torch.where(scatter_sum(vn, adj_pos[0]) > 0, 0, 1)

    return (partial_adj(adj_pos, (ucp > 0).nonzero(as_tuple=True)[0]),
            partial_adj(adj_neg, (ucn > 0).nonzero(as_tuple=True)[0]))


def test():
    return


if __name__ == "__main__":
    test()
