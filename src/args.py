from argparse import ArgumentParser


def make_args():
    parser = ArgumentParser()
    # general
    parser.add_argument('--dataset', dest='dataset', default='RND3SAT/uf250-1065',
                        type=str, help='RND3SAT DIMACS')
    parser.add_argument('--dataset_root', dest='root', default='dataset',
                        type=str, help='RND3SAT DIMACS')
    parser.add_argument('--loss', dest='loss', default='l2', type=str,
                        help='l2; cross_entropy')
    parser.add_argument('--gpu', dest='use_gpu', default=True, type=bool,
                        help='whether use gpu')
    parser.add_argument('--cuda', dest='cuda', default='0', type=str)

    # dataset
    parser.add_argument('--graph_valid_ratio', dest='graph_valid_ratio', default=0.1, type=float)
    parser.add_argument('--graph_test_ratio', dest='graph_test_ratio', default=0.1, type=float)
    parser.add_argument('--feature_transform', dest='feature_transform', type=bool, default=False,
                        help='whether pre-transform feature')

    # model
    parser.add_argument('--drop_rate', dest='drop_rate', type=float, default=0.2,
                        help='whether dropout rate, default 0.5')
    parser.add_argument('--speedup', dest='speedup', action='store_true',
                        help='whether speedup')
    parser.add_argument('--load_model', dest='load_model', default=False, type=bool,
                        help='whether load_model')
    parser.add_argument('--load_epoch', dest='load_epoch', default=0, type=int,
                        help='whether load_model')
    parser.add_argument('--batch_size', dest='batch_size', default=16,
                        type=int)  # implemented via accumulating gradient
    parser.add_argument('--num_layers', dest='num_layers', default=2, type=int)
    parser.add_argument('--num_encoder_layers', dest='num_encoder_layers', default=4, type=int)
    parser.add_argument('--num_decoder_layers', dest='num_decoder_layers', default=2, type=int)
    parser.add_argument('--num_meta_paths', dest='num_meta_paths', default=4, type=int)
    parser.add_argument('--encoder_channels', dest='encoder_channels', default='32,32,32,32', type=str)
    parser.add_argument('--decoder_channels', dest='decoder_channels', default='64,64,64,64', type=str)
    parser.add_argument('--self_att_heads', dest='self_att_heads', default=8, type=int)
    parser.add_argument('--cross_att_heads', dest='cross_att_heads', default=8, type=int)
    parser.add_argument('--heads', dest='heads', default=8, type=int)
    parser.add_argument('--activation', dest='activation', default='relu', type=str)
    # Training Setting up
    parser.add_argument('--lr', dest='lr', default=1e-6, type=float)
    parser.add_argument('--weight_decay', dest='weight_decay', default=0.01, type=float)
    parser.add_argument('--sm_par', dest='sm_par', default=5, type=float)
    parser.add_argument('--sig_par', dest='sig_par', default=15, type=float)
    parser.add_argument('--warmup_steps', dest='warmup_steps', default=4000, type=float)
    parser.add_argument('--opt_train_factor', dest='opt_train_factor', default=4, type=float)
    parser.add_argument('--epoch_num', dest='epoch_num', default=501, type=int)  # paper used: 2001
    parser.add_argument('--epoch_log', dest='epoch_log', default=50, type=int)  # test every
    parser.add_argument('--epoch_save', dest='epoch_save', default=500, type=int)  # save every
    parser.add_argument('--save_root', dest='save_root', default='saved_model', type=str)
    parser.add_argument('--save_name', dest='save_name', default='check_point', type=str)

    parser.add_argument('--in_channels', default=1, type=int)

    parser.set_defaults(gpu=True,
                        dataset='RND3SAT/uf50-218',
                        batch_size=8)  # uf100-430')
    # 'RND3SAT/uf200-860', batch_size=16)  # , load_model=True, load_epoch=500)
    args = parser.parse_args()
    return args
