from abc import ABC
import os
import torch
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_zip, extract_tar)


class SATDataset(InMemoryDataset, ABC):
    """SATDataset
    source:
        1. Uniform Random-3-SAT, phase transition region, unforced filtered
        2. DIMACS Benchmark Instances
    """
    url = "https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/{}.tar.gz"

    datasets = {
        'RND3SAT': {
            'id': ['uf50-218',
                   'uf75-325',
                   'uf100-430',
                   'uf125-538',
                   'uf150-645',
                   'uf175-753',
                   'uf200-860',
                   'uf225-960',
                   'uf250-1065'],
            'extract': extract_tar,
            'pickle': 'rnd3sat-{}.pickle',  # '1OpV4bCHjBkdpqI6H5Mg0-BqlA2ee2eBW',
        },
        'DIMACS': {
            'id': ['aim',
                   ],
            'extract': extract_tar,
            'pickle': 'dimacs-{}.pickle',   # '14FDm3NSnrBvB7eNpLeGy5Bz6FjuCSF5v',
        },
    }

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
