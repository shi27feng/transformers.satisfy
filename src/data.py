from abc import ABC

import torch
from torch_geometric.data import Dataset


class SATDataset(Dataset, ABC):
    """

    """
    def __init__(self, root, transform=None, pre_transform=None):
        super(SATDataset, self).__init__(root, transform, pre_transform)
        self.data_list = None

    def process(self):
        pass

    def len(self):
        pass

    def download(self):
        pass

    def get(self, idx):
        return
