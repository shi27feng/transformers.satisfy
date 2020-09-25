# Reference from DGL
# http://nlp.seas.harvard.edu/2018/04/03/attention.html
# https://docs.dgl.ai/en/0.4.x/tutorials/models/4_old_wines/7_transformer.html#task-and-the-dataset
import time
import torch

from tqdm import tqdm
from args import make_args
from data import SATDataset
from loss import LossCompute, LossMetric
from models import make_model
from optimizer import get_std_opt
from torch_geometric.data import DataLoader

from utils import make_checkpoint, load_checkpoint


def run_epoch(data_loader,
              model,
              loss_compute,
              device,
              args,
              is_train=True,
              desc=None,
              num_literals=None,
              num_clauses=None):
    """Standard Training and Logging Function
    Args:
        data_loader: SATDataset
        model: nn.Module
        loss_compute: function
        device: int
        is_train: bool
        desc: str
        args: dict
        num_clauses: tensor
        num_literals: tensor
    """
    # torch.autograd.set_detect_anomaly(True)
    total_loss = 0
    start = time.time()
    bs = args.batch_size
    for i, batch in tqdm(enumerate(data_loader),
                         total=len(data_loader),
                         desc=desc):
        batch = batch.to(device)
        num_lit = num_literals[i * bs, (i + 1) * bs]
        num_cls = num_clauses[i * bs, (i + 1) * bs]
        # model.encoder.reset()
        gr_idx_lit = torch.cat([torch.tensor([i] * num_lit[i]) for i in range(num_lit.size(0))])
        gr_idx_cls = torch.cat([torch.tensor([i] * num_cls[i]) for i in range(num_cls.size(0))])
        with torch.set_grad_enabled(is_train):
            adj_pos, adj_neg = batch.edge_index_pos, batch.edge_index_neg
            xv = model(batch, args)
            loss = loss_compute(xv, adj_pos, adj_neg, batch.xc.size(0), is_train)
            total_loss += loss
    elapsed = time.time() - start
    num_items = len(data_loader)
    print('average loss: {}; average time: {}'.format(total_loss / num_items, elapsed / num_items))
    return total_loss
    # TODO accuracy
    # print('accuracy: {}'.format(loss_compute.accuracy))


def main():
    args = make_args()
    device = torch.device('cuda:0') if args.use_gpu and torch.cuda.is_available() else torch.device('cpu')
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # download and save the dataset
    dataset = SATDataset(args.root, args.dataset, use_negative=False)
    dataset, perm = dataset.shuffle(return_perm=True)
    dataset.num_clauses = dataset.num_clauses[perm]
    dataset.num_literals = dataset.num_literals[perm]

    # randomly split into around 80% train, 10% val and 10% train
    last_train, last_valid = int(len(dataset) * 0.8), int(len(dataset) * 0.9)
    train_loader = DataLoader(dataset[:last_train],
                              batch_size=args.batch_size)
    valid_loader = DataLoader(dataset[last_train: last_valid],
                              batch_size=args.batch_size)
    test_loader = DataLoader(dataset[last_valid:],
                             batch_size=args.batch_size)

    # criterion = LabelSmoothing(V, padding_idx=dataset.pad_id, smoothing=0.1)
    # make_model black box
    last_epoch = 0
    model = make_model(args).to(device)
    noam_opt = get_std_opt(model, args)
    if args.load_model:
        last_epoch = args.load_epoch
        import os.path as osp
        last_epoch, loss = load_checkpoint(osp.join(args.save_root, args.save_name, '_' + last_epoch), model, noam_opt)
        print("Load Model: ", last_epoch)

    loss_metric = LossMetric()
    loss_compute = LossCompute(args.sm_par, args.sig_par, noam_opt, loss_metric.log_loss)

    for epoch in range(last_epoch, args.epoch_num + last_epoch):
        # print('Epoch: {} Training...'.format(epoch))
        model.train(True)
        total_loss = run_epoch(train_loader, model, loss_compute, device, args, is_train=True,
                               desc="Train Epoch {}".format(epoch))
        print('Epoch: {} Evaluating...'.format(epoch))
        # TODO Save model
        if epoch % args.epoch_save == 0:
            make_checkpoint(args.save_root, args.save_name, epoch, model, noam_opt.optimizer, total_loss)

        # Validation
        model.eval()
        total_loss = run_epoch(valid_loader, model, loss_compute, device, args, is_train=False,
                               desc="\t Valid Epoch {}".format(epoch))

    print('Testing...')
    model.eval()
    total_loss = run_epoch(test_loader, model, loss_compute, device, is_train=False)


if __name__ == "__main__":
    main()
