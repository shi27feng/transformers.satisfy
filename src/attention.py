import math
from abc import ABC
from typing import Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as fn
from linalg import softmax_, spmm_
from src.norms import MaskPowerNorm


class SparseAttention(nn.Module):
    """Implement the sparse scaled dot product attention with softmax.
    Inspired by:
    https://tinyurl.com/yxq4ry64 and https://tinyurl.com/yy6l47p4
    """

    def __init__(self,
                 softmax_temp=None,
                 attention_dropout=0.1):
        """
        :param heads (int):
        :param in_channels (int):
        :param softmax_temp (torch.Tensor): The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        :param attention_dropout (float): The dropout rate to apply to the attention
                           (default: 0.1)
        """
        super(SparseAttention, self).__init__()
        self.softmax_temp = softmax_temp
        self.dropout = attention_dropout

    def forward(self, queries, keys, values, adj):
        """Implements the multi-head softmax attention.
        keys and values should have same dimensions.

        Arguments
        ---------
            :param queries: torch.Tensor (N, L, E) The tensor containing the queries
            :param keys: torch.Tensor (N, S, E) The tensor containing the keys
            :param values: torch.Tensor (N, S, D) The tensor containing the values
            :param adj: the adjacency matrix plays role of mask that encodes where each query can attend to
        """
        # Extract some shapes and compute the temperature
        n, l, h, e = queries.shape  # batch, n_heads, length, depth
        m, k, s, d = values.shape

        softmax_temp = self.softmax_temp or 1. / math.sqrt(e)

        # Compute the un-normalized sparse attention according to adjacency matrix indices
        if isinstance(adj, torch.Tensor):
            adj_ = adj
            qk = torch.sum(queries.index_select(dim=-3, index=adj[0]) *
                           keys.index_select(dim=-3, index=adj[1]), dim=-1)
        else:
            qk = adj_ = None
            raise RuntimeError('currently not support non-tensor adj')

        # Compute the attention and the weighted average, adj[0] is cols idx in the same row
        alpha = fn.dropout(softmax_(softmax_temp * qk, adj[0]),
                           p=self.dropout,
                           training=self.training)
        # sparse matmul, adj as indices and qk as nonzero
        v = spmm_(adj_, alpha, l, k, values)
        v = fn.dropout(v, p=self.dropout)
        # Make sure that what we return is contiguous
        return v.contiguous()


class AddNorm(nn.Module):
    def __init__(self, normalized_shape, beta, dropout, heads, use_layer_norm=True, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        if use_layer_norm:
            self.ln = nn.LayerNorm(normalized_shape, elementwise_affine=True)
        else:
            self.ln = MaskPowerNorm(normalized_shape, group_num=heads, warmup_iters=1671 * 3)
        self.beta = beta
        if self.beta:
            self.lin_beta = nn.Linear(3 * normalized_shape, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        # self.ln.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

    def forward(self, x, y):
        if self.beta:
            b = self.lin_beta(torch.cat([y, x, y - x], dim=-1))
            b = b.sigmoid()
            return self.ln(b * x + (1 - b) * self.dropout(y))

        return self.ln(self.dropout(y) + x)


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()

    def forward(self):
        return
