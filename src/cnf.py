import os
import sys

import torch
from torch_geometric.data import Data, DataLoader


class BipartiteData(Data):
    def __init__(self, pos_adj, neg_adj, xv, xc):
        super(BipartiteData, self).__init__()
        self.edge_index_pos = pos_adj
        self.edge_index_neg = neg_adj
        self.xv = xv      # variables
        self.xc = xc      # clauses

    def __inc__(self, key, value):
        if key in ['edge_index_pos', 'edge_index_neg']:
            return torch.tensor([[self.xv.size(0)], [self.xc.size(0)]])
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
        self.edge_index_pos = [[], []]
        self.edge_index_neg = [[], []]
        self.satisfied = True
        self.text = None

    def reset(self):
        self.comments = []
        self.num_variables = 0
        self.edge_index_pos = [[], []]
        self.edge_index_neg = [[], []]
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
        Positive Clauses: {str(self.edge_index_pos)}
        Negative Clauses: {str(self.edge_index_neg)}
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
                    if literal > 0:   # positive
                        self.edge_index_pos[0].append(clause_index)
                        self.edge_index_pos[1].append(literal - 1)  # start from 0
                    else:   # negative
                        self.edge_index_neg[0].append(clause_index)
                        self.edge_index_neg[1].append(abs(literal) - 1)  # start from 0
                clause_index += 1
            else:
                break

    def to_bipartite(self, var_dim=1, cls_dim=1):
        xv = torch.empty(self.num_variables, var_dim).uniform_()
        xc = torch.ones(size=(self.num_clauses, cls_dim))
        pos_adj = torch.tensor(self.edge_index_pos)
        neg_adj = torch.tensor(self.edge_index_neg)
        return BipartiteData(pos_adj, neg_adj, xv, xc)
