from abc import ABC

import torch
import torch.nn as nn
from torch.nn.functional import mse_loss, relu
from torch_scatter import scatter


class AccuracyCompute(nn.Module, ABC):  # TODO add batch
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
    def __call__(self, xv, adj_pos, adj_neg, clause_count, gr_idx_cls, is_train):
        """
        Args:
            xv: Tensor - shape = (num_nodes, 1), e.g., [[.9], [.8], [.3], [.4]]
            adj_pos: Tensor
            adj_neg: Tensor
        Desc:
            adj[0] is an array of clause indices, adj[1] is an array of variables
        """
        sm = self._sm(xv, adj_pos, adj_neg, self.p, self.a)
        # print("XV distance: ", (xv - 0.5).square().sum() * 0.01)
        _loss = self.metric(sm, clause_count, gr_idx_cls)
        if is_train:
            _loss -= relu(10 * (sm - 0.45)).sum() * 0.005

        if self.opt is not None and is_train:
            self.opt.optimizer.zero_grad()
            _loss.backward()
            self.opt.step()

        if self.debug:
            return _loss, sm

        return _loss

    @staticmethod
    def _sm(xv, adj_pos, adj_neg, p, a):
        xv = xv.view(-1)
        # xv = LossCompute.push_to_side(xv, a)
        xn = 1 - xv
        idx = torch.cat((adj_pos[0], adj_neg[0]))
        xe = torch.cat((torch.exp(p * xv)[adj_pos[1]], torch.exp(p * xn)[adj_neg[1]]))  # exp(x*p)
        numerator = torch.mul(torch.cat((xv[adj_pos[1]], xn[adj_neg[1]])), xe)  # x*exp(x*p)
        numerator = scatter(numerator, idx, reduce="sum")
        dominator = scatter(xe, idx, reduce="sum")
        return torch.div(numerator, dominator)  # S(MAX')
        # return scatter(torch.cat((xv[adj_pos[1]], xn[adj_neg[1]])), idx, reduce='max')

    @staticmethod
    def push_to_side(x, a):  # larger a means push harder
        return 1 / (1 + torch.exp(a * (0.5 - x)))


# def flip_search(xv, xc, adj_pos, adj_neg, ori_accuracy):
#     accuracy_compute = AccuracyCompute()
#     index = torch.range(1, xc.size(0))[xc <= 0.5]
#     best_lit = None
#     best_acc = None
#     visited_lit = []
#     for cls in index:
#         pos_flip = adj_pos[1][adj_pos[0] == cls]
#         neg_flip = adj_neg[1][adj_neg[0] == cls]
#         for lit in torch.cat((pos_flip, neg_flip)):
#             if lit in visited_lit:
#                 continue
#             visited_lit.append(lit)
#             xv_ = xv.clone()
#             xv_[lit] = 1 - xv_[lit]
#             accuracy = accuracy_compute(xv_, adj_pos, adj_neg)
#             if accuracy > ori_accuracy:
#                 best_lit = lit
#                 best_acc = accuracy
#     if best_lit is not None:
#         xv[best_lit] = 1 - xv[best_lit]
#         return best_acc


class LossMetric:
    @staticmethod
    def log_loss(sm, clause_count, gr_idx_cls):
        log_smooth = torch.log(sm + 0.01)
        return -torch.sum(log_smooth)

    @staticmethod
    def linear_loss(sm, clause_count, gr_idx_cls):
        return mse_loss(sm, (torch.ones(clause_count) + 0.1).to(sm.device))

    @staticmethod
    def square_loss(sm, clause_count, gr_idx_cls):
        return (10 * (0.9 - sm)).square().sum()

    @staticmethod
    def accuracy(sm, clause_count, gr_idx_cls):
        return ((scatter(sm, gr_idx_cls, reduce="min")) // 0.500001).sum()


def literal(xi, e):
    return (1 - e) / 2 + e * xi


def negation(xi):
    return 1 - xi


def smooth_max(x, p):  # Approx Max If p is large, but will produce inf for p too large
    e = torch.exp(p * x)
    return torch.dot(x, e) / torch.sum(e)


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
