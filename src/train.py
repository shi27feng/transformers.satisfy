# DGL 
# https://docs.dgl.ai/en/0.4.x/tutorials/models/4_old_wines/7_transformer.html#task-and-the-dataset

from tqdm import tqdm
import torch as th
import numpy as np

from loss import LabelSmoothing, SimpleLossCompute
from modules import make_model
from optims import NoamOpt
from dgl.contrib.transformer import get_dataset, GraphPool

def run_epoch(data_iter, model, loss_compute, is_train=True):
    for i, g in tqdm(enumerate(data_iter)):
        with th.set_grad_enabled(is_train):
            output = model(g)
            loss = loss_compute(output, g.tgt_y, g.n_tokens)
    print('average loss: {}'.format(loss_compute.avg_loss))
    print('accuracy: {}'.format(loss_compute.accuracy))

N = 1
batch_size = 128
devices = ['cuda' if th.cuda.is_available() else 'cpu']

dataset = get_dataset("copy")
V = dataset.vocab_size
criterion = LabelSmoothing(V, padding_idx=dataset.pad_id, smoothing=0.1)
dim_model = 128

# Create model
model = make_model(V, V, N=N, dim_model=128, dim_ff=128, h=1)

# Sharing weights between Encoder & Decoder
model.src_embed.lut.weight = model.tgt_embed.lut.weight
model.generator.proj.weight = model.tgt_embed.lut.weight

model, criterion = model.to(devices[0]), criterion.to(devices[0])
model_opt = NoamOpt(dim_model, 1, 400,
                    th.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9))
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