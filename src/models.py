import copy
from abc import ABC

import torch
import torch.nn as nn

from layers import clones, LayerNorm, SublayerConnection, EncoderLayer, DecoderLayer
from torch_sparse import spspmm, transpose
from torch_geometric.utils.num_nodes import maybe_num_nodes


class Encoder(nn.Module, ABC):
    """Core encoder is a stack of N layers"""

    def __init__(self, args):
        super(Encoder, self).__init__()
        self.cached_adj = None
        self.device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
        self.cached_cls_pos_pos = None
        self.cached_cls_pos_neg = None
        self.cached_cls_neg_pos = None
        self.cached_cls_neg_neg = None

        self.cached_lit_pos_pos = None
        self.cached_lit_pos_neg = None
        self.cached_lit_neg_pos = None
        self.cached_lit_neg_neg = None

        channels = [int(n) for n in args.encoder_channels.split(',')]
        self.layers = nn.ModuleList([
            EncoderLayer(channels[i],
                         channels[i + 1],
                         channels[i + 1],
                         args.num_meta_paths,
                         args.self_att_heads,
                         args.cross_att_heads,
                         args.drop_rate) for i in range(args.num_encoder_layers)
        ])
        self.norm = LayerNorm(channels[-1])

    def forward(self, xv, xc, adj_pos, adj_neg):
        """
        meta paths are only calculated once
        """
        if self.cached_adj is None:
            self.cached_adj = [adj_pos, adj_neg]
            self._meta_paths_(adj_pos, adj_neg, device=self.device)

        meta_paths_lit = [self.cached_lit_pos_pos,   # $$A \times A^T$$
                          self.cached_lit_pos_neg,
                          self.cached_lit_neg_pos,
                          self.cached_lit_neg_neg]
        meta_paths_cls = [self.cached_cls_pos_pos,   # $$A^T \times A$$
                          self.cached_cls_pos_neg,
                          self.cached_cls_neg_pos,
                          self.cached_cls_neg_neg]
        for layer in self.layers:
            xv, xc = layer(xv, xc,
                           meta_paths_lit, meta_paths_cls,
                           adj_pos, adj_neg)
        return self.norm(xv), self.norm(xc)

    def _meta_paths_(self, adj_pos, adj_neg, device):
        # if self.cached_adj is not None:
        #     adj_pos, adj_neg = self.cached_adj
        val_pos = torch.ones(adj_pos.size(1)).to(device)
        val_neg = torch.ones(adj_neg.size(1)).to(device)
        m = max(maybe_num_nodes(adj_pos[0]), maybe_num_nodes(adj_neg[0]))
        n = max(maybe_num_nodes(adj_pos[1]), maybe_num_nodes(adj_neg[1]))
        print("edge pos: {}; edge neg: {}; m: {}; n: {}".format(adj_pos.size(1), adj_neg.size(1), m, n))
        adj_pos_t, _ = transpose(adj_pos, val_pos, m, n)
        adj_neg_t, _ = transpose(adj_neg, val_neg, m, n)

        self.cached_cls_pos_pos, self.cached_cls_pos_neg, self.cached_cls_neg_pos, self.cached_cls_neg_neg = \
            self._cross_product(adj_pos, adj_pos_t, adj_neg, adj_neg_t, val_pos, val_neg, m, n)

        self.cached_lit_pos_pos, self.cached_lit_pos_neg, self.cached_lit_neg_pos, self.cached_lit_neg_neg = \
            self._cross_product(adj_pos_t, adj_pos, adj_neg_t, adj_neg, val_pos, val_neg, n, m)

    @staticmethod
    def _cross_product(adj_p, adj_p_t, adj_n, adj_n_t, val_p, val_n, m, n):
        # cross product: $A \times A^T$
        return spspmm(adj_p, val_p, adj_p_t, val_p, m, n, m), \
               spspmm(adj_p, val_p, adj_n_t, val_n, m, n, m), \
               spspmm(adj_n, val_n, adj_p_t, val_p, m, n, m), \
               spspmm(adj_n, val_n, adj_n_t, val_n, m, n, m)


class Decoder(nn.Module, ABC):
    """Generic N layer decoder with masking."""

    def __init__(self, args):
        super(Decoder, self).__init__()
        channels = [int(n) for n in args.decoder_channels.split(',')]
        self.layers = nn.ModuleList([
            DecoderLayer(channels[i],
                         channels[i + 1],
                         channels[i + 1],
                         args.cross_att_heads,
                         args.drop_rate) for i in range(args.num_decoder_layers)
        ])

        self.norm = LayerNorm(channels[-1])
        self.last_layer = nn.Linear(channels[-1], 1)

    def forward(self, xv, xc, adj_pos, adj_neg):
        for layer in self.layers:
            xv, xc = layer(xv, xc, adj_pos, adj_neg)
        # return self.norm(xv), self.norm(xc)
        return self.last_layer(self.norm(xv))


class GraphTransformer(nn.Module, ABC):

    def __init__(self, encoder, decoder):
        super(GraphTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, xv, xc, adj_pos, adj_neg):
        # build encoders
        xv, xc = self.encode(xv, xc, adj_pos, adj_neg)
        return self.decode(xv, xc, adj_pos, adj_neg)

    def encode(self, xv, xc, adj_pos, adj_neg):
        return self.encoder(xv, xc, adj_pos, adj_neg)

    def decode(self, xv, xc, adj_pos, adj_neg):
        return self.decoder(xv, xc, adj_pos, adj_neg)


def make_model(args):
    """ Helper: Construct a model from hyper-parameters. """
    model = GraphTransformer(
        Encoder(args),
        Decoder(args))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
