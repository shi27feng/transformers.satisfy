import os
import sys

import torch
from torch_geometric.data import Data, DataLoader


class CNFData(Data):
    def __init__(self, pos_adj, neg_adj, x_s, x_t):
        super(CNFData, self).__init__()
        self.edge_index_pos = pos_adj
        self.edge_index_neg = neg_adj
        self.x_s = x_s      # variables
        self.x_t = x_t      # clauses

    def __inc__(self, key, value):
        if key in ['edge_index_pos', 'edge_index_neg']:
            return torch.tensor([[self.x_s.size(0)], [self.x_t.size(0)]])
        else:
            return super(CNFData, self).__inc__(key, value)


class CNFParser:
    def __init__(self):
        self.path = ""
        self.num_vars = 0
        self.clauses = None
        self.comments = None
        self.text = None
        self.pyg_data = None

    def read(self, path):
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
        return f"""Number of variables: {self.num_vars}
        Clauses: {str(self.clauses)}
        Comments: {str(self.comments)}"""

    def parse_dimacs(self):
        if self.text is None:
            self.read(self.path)
        self.num_vars = 0
        self.clauses = []
        self.comments = []
        occur_list = []
        n_remaining_clauses = sys.maxsize
        for line in self.text.splitlines():
            line = line.strip()
            if not line:
                continue
            elif line[0] == 'c':
                self.comments.append(line)
            elif line.startswith('p cnf'):
                tokens = line.split()
                self.num_vars, n_remaining_clauses = int(tokens[2]), \
                    min(n_remaining_clauses, int(tokens[3]))
                occur_list = [[] for _ in range(self.num_vars * 2 + 1)]
            elif n_remaining_clauses > 0:
                clause = []
                clause_index = len(self.clauses)
                for literal in line.split()[:-1]:
                    literal = int(literal)
                    clause.append(literal)
                    occur_list[literal].append(clause_index)
                self.clauses.append(clause)
                n_remaining_clauses -= 1
            else:
                break
        # return self.num_vars, clauses, occur_list, comments

    def to_pyg(self):
        pass
