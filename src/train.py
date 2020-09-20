# Reference from DGL
# http://nlp.seas.harvard.edu/2018/04/03/attention.html
# https://docs.dgl.ai/en/0.4.x/tutorials/models/4_old_wines/7_transformer.html#task-and-the-dataset
import time
import torch

from tqdm import tqdm
from args import make_args
from data import SATDataset
from loss import SimpleLossCompute2
from models import make_model
from optimizer import get_std_opt
from torch_geometric.data import DataLoader


def run_epoch(data_loader, model, loss_compute, is_train=True, desc=None):
    """Standard Training and Logging Function
    Args:
        data_loader: SATDataset
        model: nn.Module
        loss_compute: function
        device: int
        is_train: bool
        desc: str
    """
    total_loss = 0
    start = time.time()
    for i, batch in tqdm(enumerate(data_loader),
                         total=len(data_loader),
                         desc=desc):
        # batch = batch.to(device)
        with torch.set_grad_enabled(is_train):
            xv, vc = model(batch)
            adj_pos, adj_neg = batch.edge_index_pos, batch.edge_index_neg
            loss = loss_compute(xv, adj_pos, adj_neg)
            total_loss += loss
    elapsed = time.time() - start
    num_items = len(data_loader)
    print('average loss: {}; average time: {}'.format(total_loss / num_items, elapsed / num_items))
    # TODO accuracy
    # print('accuracy: {}'.format(loss_compute.accuracy))


def main():
    args = make_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # download and save the dataset
    dataset = SATDataset('dataset', 'RND3SAT/uf50-218', use_negative=False)
    dataset = dataset.to(device)
    # dataset.data = dataset.data.to(device)  # TODO need to verify
    # dataset.sat = dataset.sat.to(device)

    # randomly split into around 80% train, 10% val and 10% train
    last_train, last_valid = int(len(dataset) * 0.8), int(len(dataset) * 0.9)
    train_loader = DataLoader(dataset[:last_train],
                              batch_size=args.batch_size,
                              shuffle=True)
    valid_loader = DataLoader(dataset[last_train: last_valid],
                              batch_size=args.batch_size,
                              shuffle=True)
    test_loader = DataLoader(dataset[last_valid:],
                             batch_size=args.batch_size,
                             shuffle=True)

    # criterion = LabelSmoothing(V, padding_idx=dataset.pad_id, smoothing=0.1)
    # make_model black box
    if args.load_model:
        model = torch.load(args.save_root).to(device)
    else:
        model = make_model(args).to(device)

    opt = get_std_opt(model, args)
    loss_compute = SimpleLossCompute2(args.p, args.a, device, opt)

    for epoch in range(args.epoch_num):
        # print('Epoch: {} Training...'.format(epoch))
        model.train(True)
        run_epoch(train_loader, model, loss_compute, is_train=True,
                  desc="Train Epoch {}".format(epoch))
        print('Epoch: {} Evaluating...'.format(epoch))
        # TODO Save model
        if epoch % args.epoch_save == 0:
            torch.save(model, args.save_root)
        # Validation
        model.eval()
        run_epoch(valid_loader, model, loss_compute, is_train=False,
                  desc="\t Valid Epoch {}".format(epoch))

    print('Testing...')
    model.eval()
    run_epoch(test_loader, model, loss_compute, is_train=False)


if __name__ == "__main__":
    main()


