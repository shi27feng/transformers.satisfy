from abc import ABC
import os
import os.path as osp
from os import listdir

import torch

from cnf import CNFParser
from utils import move_to_root
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
        },
        'DIMACS': {
            'id': ['aim',
                   'jnh',
                   'ssa'],
            'extract': extract_tar,
        },
    }

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        assert self.name.split('/')[0] in self.datasets.keys()
        super(SATDataset, self).__init__(root, transform, pre_transform)
        path = osp.join(self.processed_dir, self.processed_file_names)
        self.data_list, self.sat = torch.load(path)

    @property
    def raw_file_names(self):
        return [f for f in os.listdir(self.raw_dir)]

    @property
    def processed_file_names(self):
        return '{}.pt'.format(self.name.split('/')[1])

    def download(self):
        r"""Downloads the dataset to the :obj:`self.raw_dir` folder."""
        name = self.name
        path = download_url(self.url.format(name), self.raw_dir)
        ns = self.name.split('/')
        self.datasets[ns[0]]['extract'](path, self.raw_dir)
        if ns[0] == list(self.datasets.keys())[0]:
            print('unsatisfied cases ...')
            os.unlink(path)
            name = str(ns[0] + '/u' + ns[1])
            path = download_url(self.url.format(name), self.raw_dir)
            self.datasets[ns[0]]['extract'](path, self.raw_dir)
            move_to_root(self.raw_dir)

    def process(self):
        parser = CNFParser()
        fs = listdir(self.raw_dir)
        sat = torch.ones(len(fs))
        data_list = []
        idx = 0
        for f in fs:
            parser.read(os.path.join(self.raw_dir, f))
            parser.parse_dimacs()
            data = parser.to_bipartite()
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)
            if f.startswith('uu'):
                sat[idx] = 0.0
        torch.save([data_list, sat], osp.join(self.processed_dir,
                                              self.processed_file_names))

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx], self.sat[idx]
