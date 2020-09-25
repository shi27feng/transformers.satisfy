import copy
from abc import ABC

import torch
import torch.nn as nn
from torch.nn.functional import softmax, relu
from layers import clones, SublayerConnection, EncoderLayer, DecoderLayer
from torch_sparse import spspmm, transpose
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter
from loss import LossCompute


class Encoder(nn.Module, ABC):
    """Core encoder is a stack of N layers"""

    def __init__(self, args):
        super(Encoder, self).__init__()
        self.cached_adj = None
        self.activation  = relu if args.activation == 'relu' else None
        self.device = torch.device('cuda:0') if args.use_gpu and torch.cuda.is_available() else torch.device('cpu')
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
        self.norm = nn.LayerNorm(channels[-1])

    def reset(self):
        self.cached_adj = None

    def forward(self, xv, xc, graph):
        """
        meta paths are only calculated once
        """

        meta_paths_lit = [graph.edge_index_lit_pp,   # $$A \times A^T$$
                          graph.edge_index_lit_pn,
                          graph.edge_index_lit_np,
                          graph.edge_index_lit_nn]
        meta_paths_cls = [graph.edge_index_cls_pp,   # $$A^T \times A$$
                          graph.edge_index_cls_pn,
                          graph.edge_index_cls_np,
                          graph.edge_index_cls_nn]
        for layer in self.layers:
            xv, xc = layer(xv, xc,
                           meta_paths_lit, meta_paths_cls,
                           graph.edge_index_pos, graph.edge_index_neg)
            if self.activation is not None:
                xv, xc = self.activation(xv), self.activation(xc)
                   
        return self.norm(xv), self.norm(xc)




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

        self.norm = nn.LayerNorm(channels[-1])
        self.last_layer = nn.Linear(channels[-1], 2)
        self.activation  = relu if args.activation == 'relu' else None

    def forward(self, xv, xc, graph):
        for layer in self.layers:
            xv, xc = layer(xv, xc, graph.edge_index_pos, graph.edge_index_neg)
            if self.activation is not None:
                xv, xc = self.activation(xv), self.activation(xc)

        return torch.unsqueeze(softmax(self.last_layer(self.norm(xv)), dim=1)[:, 0], 1) # First column represents closeness to 1


class GraphTransformer(nn.Module, ABC):

    def __init__(self, encoder, decoder, encoder2, decoder2):
        super(GraphTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder2 = encoder2
        self.decoder2 = decoder2

    def forward(self, graph, args):
        # build encoders
        xv, xc = self.encoder(graph.xv, graph.xc, graph)
        xv = self.decoder(xv, xc, graph)
        sm = LossCompute.get_sm(xv, graph.edge_index_pos, graph.edge_index_neg, args.sm_par, args.sig_par)
        xv, xc = self.encoder2(xv, sm.unsqueeze(1), graph)
        return self.decoder2(xv, xc, graph)

    def encode(self, xv, xc, graph):
        return self.encoder(xv, xc, graph)

    def decode(self, xv, xc, graph):
        return self.decoder(xv, xc, graph)


def make_model(args):
    """ Helper: Construct a model from hyper-parameters. """
    model = GraphTransformer(
        Encoder(args),
        Decoder(args),
        Encoder(args),
        Decoder(args))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
