from abc import ABC
import time

import torch
import torch.nn as nn
from torch_scatter import scatter

class AccuracyCompute(nn.Module):  # TODO add batch
    def __init__(self):
        super(AccuracyCompute, self).__init__()

    def __call__(self, xv, adj_pos, adj_neg, batch_size=1):
        xv = xv.view(-1)
        xv = xv // 0.5
        xp = xv[adj_pos[1]]
        xn = (1 - xv)[adj_neg[1]]
        x = torch.cat((xp, xn))
        idx = torch.cat((adj_pos[0], adj_neg[0]))
        clause_sat = scatter(x, idx, reduce="sum")
        return min(clause_sat)
        


class LabelSmoothing(nn.Module, ABC):
    def __init__(self):
        super(LabelSmoothing, self).__init__()


class SimpleLossCompute(nn.Module, ABC):
    def __init__(self, p, a, opt=None, debug=False):
        super(SimpleLossCompute, self).__init__()
        self.p = p
        self.a = a
        self.opt = opt
        self.debug = debug

    # def forward(self, xv, adj_pos, adj_neg):
    def __call__(self, xv, adj_pos, adj_neg):
        """
        Args:
            xv: Tensor - shape = (num_nodes, 1), e.g., [[.9], [.8], [.3], [.4]]
            adj_pos: Tensor
            adj_neg: Tensor
        Desc:
            adj[0] is an array of clause indices, adj[1] is an array of variables
        """
        xv = xv.view(-1)
        xp = xv[adj_pos[1]]
        xn = negation(xv[adj_neg[1]])
        x = torch.cat((xp, xn))
        xe = torch.exp(self.p * x)  # exp(x*p)
        numerator = torch.mul(x, xe)  # x*exp(x*p)
        adj = torch.cat((adj_pos, adj_neg), 1)
        idx = adj[0]
        numerator = scatter(numerator, idx, reduce="sum")
        dominator = scatter(xe, idx, reduce="sum")
        sm = push_to_side(torch.div(numerator, dominator), self.a)  # S(MAX')
        log_smooth = torch.log(sm)
        _loss = -torch.sum(log_smooth)

        if self.opt is not None:
            _loss.backward()
            self.opt.step()
            self.opt.optimizer.zero_grad()

        if self.debug:
            return _loss, sm

        return _loss


class SimpleLossCompute2(nn.Module, ABC):
    def __init__(self, p, a, opt=None, debug=False):
        super(SimpleLossCompute2, self).__init__()
        self.p = p
        self.a = a
        self.opt = opt
        self.debug = debug

    # def forward(self, xv, adj_pos, adj_neg):
    def __call__(self, xv, adj_pos, adj_neg, is_train):
        """
        Args:
            xv: Tensor - shape = (num_nodes, 1), e.g., [[.9], [.8], [.3], [.4]]
            adj_pos: Tensor
            adj_neg: Tensor
        Desc:
            adj[0] is an array of clause indices, adj[1] is an array of variables
        """
        xv = xv.view(-1)
        xn = negation(xv)
        xe = torch.cat((torch.exp(self.p * xv)[adj_pos[1]], torch.exp(self.p * xn)[adj_neg[1]]))  # exp(x*p)
        numerator = torch.mul(torch.cat((xv[adj_pos[1]],xn[adj_neg[1]])), xe)  # x*exp(x*p)
        idx = torch.cat((adj_pos[0], adj_neg[0]))
        numerator = scatter(numerator, idx, reduce="sum")
        dominator = scatter(xe, idx, reduce="sum")
        sm = push_to_side(torch.div(numerator, dominator), self.a)  # S(MAX')
        sm = sm + 0.05
        log_smooth = torch.log(sm)
        _loss = -torch.sum(log_smooth)

        if self.opt is not None and is_train:
            self.opt.optimizer.zero_grad()
            _loss.backward()
            self.opt.step()
            

        if self.debug:
            return _loss, sm

        return _loss


def literal(xi, e):
    return (1 - e) / 2 + e * xi


def negation(xi):
    return 1 - xi


def push_to_side(x, a):  # larger a means push harder
    return 1 / (1 + torch.exp(a * (0.5 - x)))


def smooth_max(x, p):  # Approx Max If p is large, but will produce inf for p too large
    exponential = torch.exp(p * x)
    return torch.dot(x, exponential) / torch.sum(exponential)


def smooth_max_test():
    # Use this function to test numerical stability of SmoothMAX with different input
    for i in range(100, 1000, 100):
        x = torch.rand(i)
        print("Length of X: ", i)
        print(torch.max(x))
        sm = smooth_max(x, 30)
        print(sm)
        print('percentile: ', 1 - sum(x > sm) / float(i))
        print("\n")


if __name__ == "__main__":
    xv = torch.tensor([[.1], [.2], [.7], [.9]])
    xv = xv.view(-1)
    edge_index_pos = torch.tensor([
        [0, 0, 1, 2],
        [0, 1, 2, 3],
    ])
    edge_index_neg = torch.tensor([
        [1, 2],
        [0, 1],
    ])

    
    
