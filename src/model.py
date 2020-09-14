from abc import ABC

import torch
import torch.nn as nn

from layers import clones, LayerNorm, SublayerConnection


class Encoder(nn.Module, ABC):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer, adj_pos, adj_neg, num_layers):
        super(Encoder, self).__init__()
        self.cached_adj = [adj_pos, adj_neg]
        self.cached_pos_pos, self.cached_pos_neg, \
            self.cached_neg_pos, self.cached_neg_neg = None, None, None, None
        self.coefficients = nn.Parameter(torch.ones(4))  # for 4 types

        self.layers = clones(layer, num_layers)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module, ABC):
    """Generic N layer decoder with masking."""

    def __init__(self, layer, num_layers):
        super(Decoder, self).__init__()
        self.layers = clones(layer, num_layers)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class GraphTransformer(nn.Module, ABC):
    def __init__(self, encoder, decoder):
        super(GraphTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, bipartite):
        # build encoders
        for i in range(self.encoder.num_layers):
            preprocess = self.encoder.preprocess(i)
            postprocess = self.encoder.postprocess(i)