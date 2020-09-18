from abc import ABC

import torch
import torch.nn as nn
from torch_scatter import scatter



class LabelSmoothing(nn.Module, ABC):
    def __init__(self):
        super(LabelSmoothing, self).__init__()


class SimpleLossCompute(nn.Module):
    def __init__(self, p, a, device):
        super(SimpleLossCompute, self).__init__()
        self.p = p
        self.a = a
        self.device = device

    def forward(self, xv, adj_pos, adj_neg):
        '''
        xv expected shape : (Nodes number, 1) i.g. [[.9], [.8], [.3], [.4]]
        adj[0] is clause, adj[1] is variable
        '''
        xv = xv.view(-1)
        # xp = literal(xv[adj_pos[1]], 1)
        xp = xv[adj_pos[1]]
        xn = negation(xv[adj_neg[1]])
        print(f"xv: {xv}")
        print(f"xp: {xp}")
        print(f"xn: {xn}")
        x = torch.cat((xp, xn))             
        xexp = torch.exp(self.p * x)         # exp(x*p) 
        numeritor = torch.mul(x, xexp)      # x*exp(x*p)
        adj = torch.cat((adj_pos, adj_neg), 1)
        idx = adj[0]
        print(f"idx : {idx}")
        print(f"numeritor: {numeritor}")
        print(f"xexp: {xexp}")
        numeritor = scatter(numeritor, idx, reduce="sum")
        dominator = scatter(xexp, idx, reduce="sum")
        print(f"numeritor: {numeritor}")
        print(f"dominator: {dominator}")
        smoothM = push_to_side(torch.div(numeritor, dominator), self.a) #S(MAX')
        print(f"smoothMax: {smoothM}")
        logsmooth = torch.log(smoothM)
        print(f"logsmoothMax: {logsmooth}")
        return -sum(logsmooth)

def literal(xi, e):
    return (1-e) / 2 + e * xi

def negation(xi):
    return 1-xi

def push_to_side(x, a): # larger a means push harder
    return 1 / (1 + torch.exp(a*(0.5 - x)))

def SmoothMAX(X, p): # Approx Max If p is large, but will produce inf for p too large
    exponential = torch.exp(p * X)
    return torch.dot(X, exponential) / torch.sum(exponential)

def SMtest():
    # Use this function to test numerical stability of SmoothMAX with different input
    for i in range(100, 1000, 100):
        x = torch.rand(i)
        print("Length of X: ", i)
        print(torch.max(x))
        SM = SmoothMAX(x, 30)
        print(SM)
        print('percentile: ', 1 - sum(x > SM) / float(i))
        print("\n")

if __name__=="__main__":
    xv = torch.tensor([[.1],[.2], [.7], [.9]])
    xv = xv.view(-1)
    edge_index_pos = torch.tensor([
        [0, 0, 1, 2],
        [0, 1, 2, 3],
    ])
    edge_index_neg = torch.tensor([
        [1, 2],
        [0, 1],
    ])
    x_s = torch.rand(4, 1)  # 2 nodes.
    x_t = torch.rand(3, 1)  # 3 nodes.
    lossFunc = SimpleLossCompute(30, 50, "cuda")
    loss = lossFunc(x_s, edge_index_pos, edge_index_neg)
    print(loss)


    
    # SMtest()
