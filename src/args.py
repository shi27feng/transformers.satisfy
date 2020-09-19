from argparse import ArgumentParser


def make_args():
    parser = ArgumentParser()
    # general
    parser.add_argument('--dataset', dest='dataset', default='grid', type=str,
                        help='grid; caveman; barabasi, cora, citeseer, pubmed')
    parser.add_argument('--loss', dest='loss', default='l2', type=str,
                        help='l2; cross_entropy')
    parser.add_argument('--gpu', dest='gpu', action='store_true',
                        type=bool, default=True,
                        help='whether use gpu')
    parser.add_argument('--cuda', dest='cuda', default='0', type=str)

    parser.add_argument('--cache_no', dest='cache', action='store_false',
                        help='whether use cache')
    # dataset
    parser.add_argument('--graph_valid_ratio', dest='graph_valid_ratio', default=0.1, type=float)
    parser.add_argument('--graph_test_ratio', dest='graph_test_ratio', default=0.1, type=float)
    parser.add_argument('--feature_transform', dest='feature_transform', action='store_true',
                        help='whether pre-transform feature')

    # model
    parser.add_argument('--drop_rate', dest='drop_rate', action='store_true',
                        help='whether dropout rate, default 0.5')
    parser.add_argument('--speedup', dest='speedup', action='store_true',
                        help='whether speedup')
    parser.add_argument('--load_model', dest='load_model', action='store_true',
                        help='whether load_model')
    parser.add_argument('--batch_size', dest='batch_size', default=64,
                        type=int)  # implemented via accumulating gradient
    parser.add_argument('--num_layers', dest='num_layers', default=2, type=int)
    parser.add_argument('--num_encoders', dest='num_encoder_layers', default=2, type=int)
    parser.add_argument('--num_decoders', dest='num_decoder_layers', default=2, type=int)
    parser.add_argument('--in_channels', dest='in_channels', default=1, type=int)
    parser.add_argument('--out_channels', dest='out_channels', default=32, type=int)

    parser.add_argument('--lr', dest='lr', default=1e-3, type=float)
    parser.add_argument('--epoch_num', dest='epoch_num', default=201, type=int)   # paper used: 2001
    parser.add_argument('--epoch_log', dest='epoch_log', default=50, type=int)  # test every
    parser.add_argument('--epoch_save', dest='epoch_save', default=50, type=int)  # save every
    parser.add_argument('--save_root', dest='save_root', default='saved_model', type=str)

    parser.set_defaults(gpu=True, task='link', model='GCN', dataset='Cora', cache=True,
                        feature_pre=True, dropout=False, recompute_template=False, load_model=False,
                        speedup=False)
    args = parser.parse_args()
    return args
