# DGL 
# https://docs.dgl.ai/en/0.4.x/tutorials/models/4_old_wines/7_transformer.html#task-and-the-dataset

from tqdm import tqdm
import torch as th
import numpy as np
import random

from args import make_args
from loss import LabelSmoothing, SimpleLossCompute
from models import make_model
from optimizer import get_std_opt
from data import SATDataset
from torch_geometric. data import DataLoader


def run_epoch(data_loader, model, loss_compute, is_train=True, desc=None):
    for i, g in tqdm(enumerate(data_loader),
                     total=len(data_loader),
                     desc=desc):
        with th.set_grad_enabled(is_train):
            output = model(g)
            loss = loss_compute(output)
    print('average loss: {}'.format(loss_compute.avg_loss))
    print('accuracy: {}'.format(loss_compute.accuracy))


def save_model(model, root):
    pass


def main():
    args = make_args()
    devices = ['cuda' if th.cuda.is_available() else 'cpu']

    # download and save the dataset
    dataset = SATDataset('dataset', 'RND3SAT/uf50-218')

    # randomly split into around 80% train, 10% val and 10% train
    last_train, last_valid = int(len(dataset)*0.8), int(len(dataset) * 0.9)
    train_loader = DataLoader(dataset[:last_train],
                              batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset[last_train: last_valid],
                              batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset[last_valid:],
                             batch_size=args.batch_size, shuffle=True)

    # criterion = LabelSmoothing(V, padding_idx=dataset.pad_id, smoothing=0.1)
    # make_model black box
    model = make_model(args)
    model_opt = get_std_opt(model)
    loss_compute = SimpleLossCompute

    att_maps = []
    for epoch in range(args.epoch_num):
        # print('Epoch: {} Training...'.format(epoch))
        model.train(True)
        run_epoch(train_loader, model,
                  loss_compute(),
                  is_train=True, desc="Train Epoch {}".format(epoch))
        print('Epoch: {} Evaluating...'.format(epoch))
        # TODO Save model
        if epoch % args.epoch_save == 0:
            save_model(model, args.save_root)
        model.eval()
        run_epoch(valid_loader, model,
                  loss_compute(),
                  is_train=False, desc="Valid Epoch {}".format(epoch))
        att_maps.append(model.att_weight_map)
        print('Epoch: {} Testing...'.format(epoch))
        model.att_weight_map = None
        model.eval()
        run_epoch(test_loader, model,
                  loss_compute(), is_train=False)


if __name__ == "__main__":
    main()


