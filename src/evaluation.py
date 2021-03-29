# Reference from DGL
# http://nlp.seas.harvard.edu/2018/04/03/attention.html
# https://docs.dgl.ai/en/0.4.x/tutorials/models/4_old_wines/7_transformer.html#task-and-the-dataset
import torch
from torch_geometric.data import DataLoader

from args import make_args
from data import SatDataset
from loss import LossCompute, LossMetric
from models import make_model
from optimizer import get_std_opt
from train import run_epoch
from utils import load_checkpoint


def main():
    # torch.cuda.empty_cache()
    args = make_args()
    device = torch.device('cuda:0') if args.use_gpu and torch.cuda.is_available() else torch.device('cpu')
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # download and save the dataset
    dataset = SatDataset(args.root, args.dataset, use_negative=False)
    dataset, perm = dataset.shuffle(return_perm=True)
    num_clauses = dataset.num_clauses[perm]
    num_literals = dataset.num_literals[perm]

    # randomly split into around 80% train, 10% val and 10% train
    last_train, last_valid = int(len(dataset) * 0.8), int(len(dataset) * 0.9)
    test_loader = DataLoader(dataset[last_valid:],
                             batch_size=args.batch_size)

    # criterion = LabelSmoothing(V, padding_idx=dataset.pad_id, smoothing=0.1)
    # make_model black box
    last_epoch = args.load_epoch
    model = make_model(args).to(device)
    noam_opt = get_std_opt(model, args)

    import os.path as osp
    last_epoch, loss = load_checkpoint(osp.join(args.save_root,
                                                args.save_name + '_' + str(last_epoch) + '.pickle'),
                                       model, noam_opt.optimizer)
    print("Load Model: ", last_epoch)

    loss_metric = LossMetric()
    accuracy_compute = LossCompute(args.sm_par, args.sig_par, noam_opt, loss_metric.accuracy, debug=True)

    print('Testing...')
    model.eval()
    _, sat_test = run_epoch(test_loader, model, accuracy_compute, device, args, is_train=False,
                            num_literals=num_literals[last_valid:], num_clauses=num_clauses[last_valid:])


if __name__ == "__main__":
    main()
