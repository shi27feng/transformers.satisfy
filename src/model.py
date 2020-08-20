from abc import ABC

import torch.nn as nn

from layers import clones


class MultiHeadAttention(nn.Module, ABC):
    """Multi-Head Attention"""
    def __init__(self, h, dim_model):
        """
        Args:
            h: number of heads;
            dim_model: hidden dimension
        """
        super(MultiHeadAttention, self).__init__()
        self.d_k = dim_model // h
        self.h = h
        # W_q, W_k, W_v, W_o
        self.linears = clones(nn.Linear(dim_model, dim_model), 4)

    def get(self, x, fields='qkv'):
        """
        Returns: dict, a dict of queries / keys / values.
        """
        batch_size = x.shape[0]
        ret = {}
        if 'q' in fields:
            ret['q'] = self.linears[0](x).view(batch_size, self.h, self.d_k)
        if 'k' in fields:
            ret['k'] = self.linears[1](x).view(batch_size, self.h, self.d_k)
        if 'v' in fields:
            ret['v'] = self.linears[2](x).view(batch_size, self.h, self.d_k)
        return ret

    def readout(self, x):
        """get output of the multi-head attention"""
        batch_size = x.shape[0]
        return self.linears[3](x.view(batch_size, -1))

