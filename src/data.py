from abc import ABC
import os
import torch
from torch_geometric.data import Dataset


class SATDataset(Dataset, ABC):
    """SATDataset

    """
    def __init__(self, root, transform=None, pre_transform=None):
        super(SATDataset, self).__init__(root, transform, pre_transform)
        self.data_list = None

    @property
    def raw_file_names(self):
        return [f for f in os.listdir(self.raw_dir)]

    @property
    def processed_file_names(self):
        r"""The name of the files to find in the :obj:`self.processed_dir`
        folder in order to skip the processing."""
        raise NotImplementedError

    def download(self):
        r"""Downloads the dataset to the :obj:`self.raw_dir` folder."""
        raise NotImplementedError

    def process(self):
        r"""Processes the dataset to the :obj:`self.processed_dir` folder."""
        raise NotImplementedError

    def len(self):
        raise NotImplementedError

    def get(self, idx):
        r"""Gets the data object at index :obj:`idx`."""
        raise NotImplementedError
