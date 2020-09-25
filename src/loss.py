from abc import ABC
import time

import torch
import torch.nn as nn
from torch_scatter import scatter
from torch.nn.functional import mse_loss

class AccuracyCompute(nn.Module):  # TODO add batch
    def __init__(self):
        super(AccuracyCompute, self).__init__()

    def __call__(self, xv, adj_pos, adj_neg, batch_size=1):
        xv = xv.view(-1)
        xv = xv // 0.50001
        xp = xv[adj_pos[1]]
        xn = (1 - xv)[adj_neg[1]]
        x = torch.cat((xp, xn))
        idx = torch.cat((adj_pos[0], adj_neg[0]))
        clause_sat = scatter(x, idx, reduce="sum")
        return min(clause_sat)
        


class LabelSmoothing(nn.Module, ABC):
    def __init__(self):
        super(LabelSmoothing, self).__init__()

class LossCompute(nn.Module, ABC):
    def __init__(self, p, a, opt=None, metric=None, debug=False):
        super(LossCompute, self).__init__()
        self.p = p
        self.a = a
        self.opt = opt
        self.debug = debug
        self.metric = metric

    # def forward(self, xv, adj_pos, adj_neg):
    def __call__(self, xv, adj_pos, adj_neg, clause_count, is_train):
        """
        Args:
            xv: Tensor - shape = (num_nodes, 1), e.g., [[.9], [.8], [.3], [.4]]
            adj_pos: Tensor
            adj_neg: Tensor
        Desc:
            adj[0] is an array of clause indices, adj[1] is an array of variables
        """     
        sm = self.get_sm(xv, adj_pos, adj_neg, self.p, self.a)
        _loss = self.metric(sm, clause_count)

        if self.opt is not None and is_train:
            self.opt.optimizer.zero_grad()
            _loss.backward()
            self.opt.step()
            

        if self.debug:
            return _loss, sm

        return _loss

    @staticmethod
    def get_sm(xv, adj_pos, adj_neg, p, a):
        xv = xv.view(-1)
        xn = 1 - xv
        xe = torch.cat((torch.exp(p * xv)[adj_pos[1]], torch.exp(p * xn)[adj_neg[1]]))  # exp(x*p)
        numerator = torch.mul(torch.cat((xv[adj_pos[1]],xn[adj_neg[1]])), xe)  # x*exp(x*p)
        idx = torch.cat((adj_pos[0], adj_neg[0]))
        numerator = scatter(numerator, idx, reduce="sum")
        dominator = scatter(xe, idx, reduce="sum")
        return LossCompute.push_to_side(torch.div(numerator, dominator), a)  # S(MAX')

    @staticmethod
    def push_to_side(x, a):  # larger a means push harder
        return 1 / (1 + torch.exp(a * (0.5 - x)))

    

class LossMetric():
    @staticmethod
    def log_loss(sm, clause_count):
        log_smooth = torch.log(sm + 0.05)
        return -torch.sum(log_smooth)
    @staticmethod
    def linear_loss(sm, clause_count):
        return mse_loss(sm, (torch.ones(clause_count) + 0.1).to(sm.device))
    @staticmethod
    def square_loss(sm, clause_count):
        return (10*(1 - sm)).square().sum()





def literal(xi, e):
    return (1 - e) / 2 + e * xi


def negation(xi):
    return 1 - xi

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

    
    
