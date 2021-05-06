from abc import ABC

import torch
import torch.nn as nn
from torch.nn.functional import relu, sigmoid

from attentions import EncoderLayer, DecoderLayer


class GraphTransformer(nn.Module, ABC):

    def __init__(self, args):
        super(GraphTransformer, self).__init__()
        self.args = args
        channels = [int(n) for n in args.decoder_channels.split(',')]
        self.emb_c = nn.Linear(in_features=args.in_channels, out_features=channels[0])
        self.emb_v = nn.Linear(in_features=args.in_channels, out_features=channels[0])
        self.enc_ = nn.ModuleList([
            EncoderLayer(in_channels=channels[i],
                         hd_channels=channels[i + 1],
                         heads=args.heads,
                         num_meta_paths=args.num_meta_paths,
                         dropout=args.drop_rate) for i in range(len(channels) - 1)
        ])
        self.dec_ = nn.ModuleList([
            DecoderLayer(in_channels=channels[i],
                         hd_channels=channels[i + 1],
                         heads=args.heads,
                         dropout=args.drop_rate) for i in range(len(channels) - 1)
        ])
        
    def forward(self, graph):
        meta_paths_var = [graph.edge_index_var_pp,  # $$A \times A^T$$
                          graph.edge_index_var_pn,
                          graph.edge_index_var_np,
                          graph.edge_index_var_nn]
        meta_paths_cls = [graph.edge_index_cls_pp,  # $$A^T \times A$$
                          graph.edge_index_cls_pn,
                          graph.edge_index_cls_np,
                          graph.edge_index_cls_nn]
        # build encoders
        x, c, adj_pos, adj_neg = graph.xv, graph.xc, graph.edge_index_pos, graph.edge_index_neg
        x = self.emb_v(x)
        c = self.emb_c(c)
        for i in range(len(self.enc_)):
            x, c = self.enc_[i](x, c, meta_paths_var, meta_paths_cls)
            x, c = self.dec_[i](x, c, adj_pos, adj_neg)
        return torch.sigmoid(x)


def make_model(args):
    """ Helper: Construct a model from hyper-parameters. """
    model = GraphTransformer(args)

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
