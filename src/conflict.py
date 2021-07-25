import torch


def unsatisfied_clauses(clauses: torch.Tensor):
    return torch.nonzero(clauses, as_tuple=True)[0]


def vars_of_clauses(clauses, adj):
    rows, cols = adj  # rows -> cls, cols -> var
    s = torch.cat([torch.tensor([0]),
                   torch.nonzero(rows[1:] - rows[:-1], as_tuple=True)[0] + 1,
                   torch.tensor([len(cols)])])
    return torch.unique(torch.cat([cols[s[i]: s[i + 1]] for i in clauses]),
                        sorted=True)


def intersection(t, s):
    u, c = torch.unique(torch.cat([t, s]), return_counts=True)
    return u[c > 1]


def conflict_clauses(clauses, vars, adj):
    # get the list of solved clauses
    sc = torch.nonzero(clauses, as_tuple=True)[0]
    v = torch.zeros(torch.max(adj[0]).item() + 1)
    v[vars] = 1.
    c = torch.sparse_coo_tensor(adj, torch.ones(torch.max(adj[0]).item() + 1)) @ v
    return intersection(sc, torch.nonzero(c, as_tuple=True)[0])


def test():
    return


if __name__ == "__main__":
    test()
