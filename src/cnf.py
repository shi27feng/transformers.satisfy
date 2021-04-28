import os
import re
import sys

import torch
from torch_geometric.data import Data
from torch_sparse import spspmm
from linalg import transpose_


class BipartiteData(Data):
    def __init__(self, pos_adj, neg_adj, xv, xc):
        super(BipartiteData, self).__init__()
        self.edges_pos = pos_adj
        self.edges_neg = neg_adj
        self.xv = xv  # variables
        self.xc = xc  # clauses

        self.edges_var_pp = self.edges_var_pn = self.edges_var_np = self.edges_var_nn = None
        self.edges_cls_pp = self.edges_cls_pn = self.edges_cls_np = self.edges_cls_nn = None

        self._meta_paths_(pos_adj, neg_adj)
        self._put_back_cpu()

    @staticmethod
    def _cross_product(adj_p, adj_p_t, adj_n, adj_n_t, val_p, val_n, m, n):
        # cross product: $A \times A^T$
        return spspmm(adj_p, val_p, adj_p_t, val_p, m, n, m)[0], \
               spspmm(adj_p, val_p, adj_n_t, val_n, m, n, m)[0], \
               spspmm(adj_n, val_n, adj_p_t, val_p, m, n, m)[0], \
               spspmm(adj_n, val_n, adj_n_t, val_n, m, n, m)[0]

    def _meta_paths_(self, adj_pos, adj_neg):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        adj_pos = adj_pos.to(device)
        adj_neg = adj_neg.to(device)
        # print("edge pos: {}; edge neg: {}; m: {}; n: {}".format(adj_pos.size(1), adj_neg.size(1), m, n))
        adj_pos_t, _ = transpose_(adj_pos)
        adj_neg_t, _ = transpose_(adj_neg)

        self.edges_cls_pp, self.edges_cls_pn, self.edges_cls_np, self.edges_cls_nn = \
            self._cross_product(adj_pos, adj_pos_t, adj_neg, adj_neg_t, val_pos, val_neg, m, n)

        self.edges_var_pp, self.edges_var_pn, self.edges_var_np, self.edges_var_nn = \
            self._cross_product(adj_pos_t, adj_pos, adj_neg_t, adj_neg, val_pos, val_neg, n, m)

    def _put_back_cpu(self):
        self.edges_var_pp = self.edges_var_pp.to("cpu")
        self.edges_var_pn = self.edges_var_pn.to("cpu")
        self.edges_var_np = self.edges_var_np.to("cpu")
        self.edges_var_nn = self.edges_var_nn.to("cpu")

        self.edges_cls_pp = self.edges_cls_pp.to("cpu")
        self.edges_cls_pn = self.edges_cls_pn.to("cpu")
        self.edges_cls_np = self.edges_cls_np.to("cpu")
        self.edges_cls_nn = self.edges_cls_nn.to("cpu")

    def __inc__(self, key, value):
        if bool(re.search('(pos|neg)', key)):
            return torch.tensor([[self.xc.size(0)], [self.xv.size(0)]])
        elif bool(re.search('var', key)):
            return torch.tensor([[self.xv.size(0)], [self.xv.size(0)]])
        elif bool(re.search('cls', key)):
            return torch.tensor([[self.xc.size(0)], [self.xc.size(0)]])
        else:
            return super(BipartiteData, self).__inc__(key, value)

    @property
    def num_node_features(self):
        return self.xv.size(1), self.xc.size(1)

    # @property   # TODO this one causes problem
    # def num_nodes(self):
    #     return self.xv.size(0), self.xc.size(0)


class CNFParser:
    def __init__(self):
        self.path = ""
        self.comments = []
        self.num_clauses = 0
        self.num_variables = 0
        self.edges_pos = [[], []]  # [clauses, variables]
        self.edges_neg = [[], []]
        self.satisfied = True
        self.text = None

    def reset(self):
        self.comments = []
        self.num_variables = 0
        self.edges_pos = [[], []]
        self.edges_neg = [[], []]
        self.text = None

    def read(self, path):
        self.reset()
        if path is None or path == '':
            raise ValueError("path can't be empty")
        if os.path.exists(path):
            self.path = path
        else:
            raise IOError("file [", path, "] doesn't exist.")
        with open(self.path, 'r') as f:
            try:
                self.text = f.read()
            except IOError:
                print("Can't read file: ", self.path)

    def __str__(self):
        return f"""Number of variables: {self.num_variables}
        Positive Clauses: {str(self.edges_pos)}
        Negative Clauses: {str(self.edges_neg)}
        Comments: {str(self.comments)}"""

    def parse_dimacs(self):
        if self.text is None:
            self.read(self.path)
        n_remaining_clauses = sys.maxsize
        clause_index = 0
        for line in self.text.splitlines():
            line = line.strip()
            if not line:
                continue
            elif line[0] == 'c':
                if "Not Satisfiable" in line:
                    self.satisfied = False
                self.comments.append(line)
            elif line.startswith('p cnf'):
                tokens = line.split()
                self.num_variables, n_remaining_clauses = int(tokens[2]), \
                    min(n_remaining_clauses, int(tokens[3]))
                self.num_clauses = n_remaining_clauses
            elif clause_index < n_remaining_clauses:
                for literal in line.split()[:-1]:
                    literal = int(literal)
                    if literal > 0:  # positive
                        self.edges_pos[0].append(clause_index)
                        self.edges_pos[1].append(literal - 1)  # start from 0
                    else:  # negative
                        self.edges_neg[0].append(clause_index)
                        self.edges_neg[1].append(abs(literal) - 1)  # start from 0
                clause_index += 1
            else:
                break

    def to_bipartite(self, var_dim=1, cls_dim=1):
        # xv = torch.empty(self.num_variables, var_dim).uniform_()
        xv = torch.ones(size=(self.num_variables, var_dim))
        xc = torch.ones(size=(self.num_clauses, cls_dim)) * 0.5
        pos_adj = torch.tensor(self.edges_pos)
        neg_adj = torch.tensor(self.edges_neg)
        return BipartiteData(pos_adj, neg_adj, xv, xc)
