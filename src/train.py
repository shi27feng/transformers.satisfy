# DGL 
# https://docs.dgl.ai/en/0.4.x/tutorials/models/4_old_wines/7_transformer.html#task-and-the-dataset

from tqdm import tqdm
import torch as th
import numpy as np
import random

from loss import LabelSmoothing, SimpleLossCompute
from models import make_model
from optimizer import get_std_opt
from data import SATDataset

def run_epoch(data_iter, model, loss_compute, is_train=True):
    for i, g in tqdm(enumerate(data_iter)):
        with th.set_grad_enabled(is_train):
            output = model(g)
            loss = loss_compute(output, g.tgt_y, g.n_tokens)
    print('average loss: {}'.format(loss_compute.avg_loss))
    print('accuracy: {}'.format(loss_compute.accuracy))


def main():
    N = 1
    n_batch_size = 128
    devices = ['cuda' if th.cuda.is_available() else 'cpu']

    # download and save the dataset
    dataset = SATDataset('dataset', 'RND3SAT/uf50-218')
    # load the dataset
    sat_dataloader = DataLoader(dataset, batch_size=n_batch_size, shuffle=True)
    # randomly split into around 80% train, 10% val and 10% train
    train_iter = []
    val_iter = []
    test_iter = []
    for batch in (ntu_dataloader):
        i = random.randint(1, 10)
        if i <= 8:
            train_iter.append(batch)
        else if i == 9:
            val_iter.append(batch)
        else:
            test_iter.append(batch)

    criterion = LabelSmoothing(V, padding_idx=dataset.pad_id, smoothing=0.1)
    dim_model = 128
    # make_model black box
    model = make_model(V, V, N=N, dim_model=128, dim_ff=128, h=1)

    # Sharing weights between Encoder & Decoder
    model.src_embed.lut.weight = model.tgt_embed.lut.weight
    model.generator.proj.weight = model.tgt_embed.lut.weight

    model, criterion = model.to(devices[0]), criterion.to(devices[0])
    model_opt = get_std_opt(model)
    loss_compute = SimpleLossCompute

    att_maps = []
    for epoch in range(4):
        train_iter = dataset(graph_pool, mode='train', batch_size=batch_size, devices=devices)
        valid_iter = dataset(graph_pool, mode='valid', batch_size=batch_size, devices=devices)
        print('Epoch: {} Training...'.format(epoch))
        model.train(True)
        run_epoch(train_iter, model,
                loss_compute(criterion, model_opt), is_train=True)
        print('Epoch: {} Evaluating...'.format(epoch))
        model.att_weight_map = None
        model.eval()
        run_epoch(valid_iter, model,
                loss_compute(criterion, None), is_train=False)
        att_maps.append(model.att_weight_map)
        print('Epoch: {} Testing...'.format(epoch))
        model.att_weight_map = None
        model.eval()
        run_epoch(test_iter, model,
                loss_compute(criterion, None), is_train=False)


if __name__ == "__main__":
    main()


