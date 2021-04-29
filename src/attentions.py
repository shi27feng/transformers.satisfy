import math
from abc import ABC
from typing import Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as fn
from linalg import softmax_, spmm_, transpose_


class SparseAttention(nn.Module):
    """Implement the sparse scaled dot product attention with softmax.
    Inspired by:
    https://tinyurl.com/yxq4ry64 and https://tinyurl.com/yy6l47p4
    """

    def __init__(self,
                 softmax_temp=None,
                 dropout=0.1):
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
        self.dropout = dropout

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
        l, h, e = queries.shape[-3:]  # length, heads, depth
        k, _, _ = values.shape[-3:]

        softmax_temp = self.softmax_temp or 1. / math.sqrt(e)

        # Compute the un-normalized sparse attention according to adjacency matrix indices
        qk = torch.sum(queries.index_select(dim=-3, index=adj[0]) *
                       keys.index_select(dim=-3, index=adj[1]), dim=-1)

        # Compute the attention and the weighted average, adj[0] is cols idx in the same row
        alpha = softmax_(softmax_temp * qk, adj[0])
        # sparse matmul, adj as indices and qk as nonzero
        v = spmm_(adj, alpha, l, k, values)
        v = fn.dropout(v, p=self.dropout)
        # Make sure that what we return is contiguous
        return v.contiguous()


class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, x, y):
        return self.ln(self.dropout(y) + x)


class FeedForward(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, in_channels),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 hd_channels,
                 heads,
                 num_meta_paths,
                 dropout):
        super(EncoderLayer, self).__init__()
        self.path_weight_var = nn.Parameter(torch.ones(num_meta_paths))
        self.path_weight_cls = nn.Parameter(torch.ones(num_meta_paths))

        self.lin_qkv_var = nn.Linear(in_channels, hd_channels * 3)
        self.lin_qkv_cls = nn.Linear(in_channels, hd_channels * 3)
        self.lin_kv = nn.Linear(in_channels, hd_channels * 2)
        self.lin_q = nn.Linear(in_channels, hd_channels)

        self.mha = SparseAttention(dropout=dropout)

        self.add_norm_att_var = AddNorm(hd_channels, dropout)
        self.add_norm_ffn_var = AddNorm(hd_channels, dropout)
        self.ffn_var = FeedForward(hd_channels, hd_channels, dropout)

        self.add_norm_att_cls = AddNorm(hd_channels, dropout)
        self.add_norm_ffn_cls = AddNorm(hd_channels, dropout)
        self.ffn_cls = FeedForward(hd_channels, hd_channels, dropout)

    @staticmethod
    def _attention_meta_path(x, lin_qkv, meta_paths, attn, path_weights):
        q, k, v = lin_qkv(x).chunk(3, dim=-1)
        res = torch.zeros_like(x)
        for i in range(len(meta_paths)):  # they are not sequential, but in reduction mode
            res += path_weights[i] * attn(q, k, v, meta_paths[i])  # TODO
        return res

    @staticmethod
    def _cross_attention(x, y, lin_q, lin_k, attn, adj):
        adj_ = transpose_(adj)

        return

    def forward(self, v, c, meta_paths_var, meta_paths_cls, adj_pos, adj_neg):
        v_ = self._attention_meta_path(v, self.lin_qkv_var, meta_paths_var, self.mha, self.path_weight_var)
        c_ = self._attention_meta_path(c, self.lin_qkv_cls, meta_paths_cls, self.mha, self.path_weight_cls)

        return
