import copy
from abc import ABC

import torch
import torch.nn as nn

from layers import clones, LayerNorm, SublayerConnection, EncoderLayer, DecoderLayer
from torch_sparse import spspmm, transpose
from torch_geometric.utils.num_nodes import maybe_num_nodes


class Encoder(nn.Module, ABC):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer, adj_pos, adj_neg, num_layers):
        super(Encoder, self).__init__()
        self.cached_adj = [adj_pos, adj_neg]

        self.cached_cls_pos_pos = None
        self.cached_cls_pos_neg = None
        self.cached_cls_neg_pos = None
        self.cached_cls_neg_neg = None

        self.cached_lit_pos_pos = None
        self.cached_lit_pos_neg = None
        self.cached_lit_neg_pos = None
        self.cached_lit_neg_neg = None

        self.layers = clones(layer, num_layers)
        self.norm = LayerNorm(layer.size)
        self._meta_paths_(adj_pos=adj_pos, adj_neg=adj_neg)

    def forward(self, xv, xc):
        """Pass the input (and mask) through each layer in turn."""
        meta_paths_lit = [self.cached_lit_pos_pos,
                          self.cached_lit_pos_neg,
                          self.cached_lit_neg_pos,
                          self.cached_lit_neg_neg]
        meta_paths_cls = [self.cached_cls_pos_pos,
                          self.cached_cls_pos_neg,
                          self.cached_cls_neg_pos,
                          self.cached_cls_neg_neg]
        for layer in self.layers:
            x = layer(xv,
                      xc,
                      meta_paths_lit, meta_paths_cls,
                      self.adj_pos, self.adj_neg)
        return self.norm(xv), self.norm(xc)

    def _meta_paths_(self, adj_pos, adj_neg):
        if self.cached_adj is not None:
            adj_pos, adj_neg = self.cached_adj
        val_pos = torch.ones(adj_pos.size(1))
        val_neg = torch.ones(adj_neg.size(1))
        m = maybe_num_nodes(adj_pos[0], adj_neg[0])
        n = maybe_num_nodes(adj_pos[1], adj_neg[1])
        adj_pos_t, _ = transpose(adj_pos, val_pos, m, n)
        adj_neg_t, _ = transpose(adj_neg, val_pos, m, n)

        self.cached_cls_pos_pos, self.cached_cls_pos_neg, self.cached_cls_neg_pos, self.cached_cls_neg_neg = \
            self._cross_product(adj_pos, adj_pos_t, adj_neg, adj_neg_t, val_pos, val_neg, m, n)

        self.cached_lit_pos_pos, self.cached_lit_pos_neg, self.cached_lit_neg_pos, self.cached_lit_neg_neg = \
            self._cross_product(adj_pos_t, adj_pos, adj_neg_t, adj_neg, val_pos, val_neg, n, m)

    @staticmethod
    def _cross_product(adj_pos, adj_pos_t, adj_neg, adj_neg_t, val_pos, val_neg, m, n):
        # cross product: $A \times A^T$
        return spspmm(adj_pos, val_pos, adj_pos_t, val_pos, m, n, m), \
               spspmm(adj_pos, val_pos, adj_neg_t, val_neg, m, n, m), \
               spspmm(adj_neg, val_neg, adj_pos_t, val_pos, m, n, m), \
               spspmm(adj_neg, val_neg, adj_neg_t, val_neg, m, n, m)


class Decoder(nn.Module, ABC):
    """Generic N layer decoder with masking."""

    def __init__(self, layer, num_layers):
        super(Decoder, self).__init__()
        self.layers = clones(layer, num_layers)
        self.norm = LayerNorm(layer.size)

    def forward(self, xv, xc, adj_pos, adj_neg):
        for layer in self.layers:
            x = layer(xv, xc, adj_pos, adj_neg)
        return self.norm(xv), self.norm(xc)


class GraphTransformer(nn.Module, ABC):

    def __init__(self, encoder, decoder):
        super(GraphTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, xv, xc, adj_pos, adj_neg):
        # build encoders
        xv, xc = self.encode(xv, xc)
        return self.decode(xv, xc, adj_pos, adj_neg)

    def encode(self, xv, xc):
        return self.encoder(xv, xc)

    def decode(self, xv, xc, adj_pos, adj_neg):
        return self.decoder(xv, xc, adj_pos, adj_neg)


def make_model(xv, xc, adj_pos, adj_neg, args):
    """Helper: Construct a model from hyperparameters."""
    c = copy.deepcopy
    ff = nn.Linear(args.in_channels, args.out_channels)
    model = GraphTransformer(
        Encoder(EncoderLayer(args), adj_pos, adj_neg, args.num_layers),
        Decoder(DecoderLayer(args, c(ff)), args.num_layers))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model
