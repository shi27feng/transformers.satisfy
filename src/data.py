from abc import ABC
import os
import os.path as osp
from os import listdir
from tqdm import tqdm
from time import sleep

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

    def __init__(self, root, name, use_negative, transform=None, pre_transform=None):
        self.name = name
        self.use_negative = use_negative
        assert self.name.split('/')[0] in self.datasets.keys()
        if not osp.exists(osp.join(root, "raw")):
            os.mkdir(osp.join(root, "raw"))
        super(SATDataset, self).__init__(root, transform, pre_transform)
        path = osp.join(self.processed_dir, self.processed_file_names)
        self.data, self.sat, self.num_literals, self.num_clauses = torch.load(path)

    @property
    def raw_file_names(self):
        return [f for f in os.listdir(self.raw_dir)]

    @property
    def processed_file_names(self):
        return '{}.pt'.format(self.name.split('/')[1])

    @property
    def num_node_features(self):
        r"""Returns the number of features per node in the dataset."""
        return self.data[0].xv.num_node_features, \
            self.data[0].xc.num_node_features

    def download(self):
        r"""Downloads the dataset to the :obj:`self.raw_dir` folder."""
        print('Satisfied Cases ...')
        name = self.name
        path = download_url(self.url.format(name), self.raw_dir)
        ns = self.name.split('/')
        self.datasets[ns[0]]['extract'](path, self.raw_dir)
        os.unlink(path)
        if self.use_negative and ns[0] == list(self.datasets.keys())[0]:
            print('Unsatisfied Cases ...')
            name = str(ns[0] + '/u' + ns[1])
            path = download_url(self.url.format(name), self.raw_dir)
            self.datasets[ns[0]]['extract'](path, self.raw_dir)
            move_to_root(self.raw_dir)
            os.unlink(path)

    def process(self):
        parser = CNFParser()
        fs = listdir(self.raw_dir)
        progress_bar = tqdm(fs)
        sat = torch.ones(len(fs))
        data_list, num_literals, num_clauses = [], [], []
        idx = 0
        for f in progress_bar:
            sleep(1e-4)
            progress_bar.set_description("processing %s" % f)
            path = os.path.join(self.raw_dir, f)
            parser.read(path)
            parser.parse_dimacs()
            data = parser.to_bipartite()
            num_literals.append(data.xv.size(0))
            num_clauses.append(data.xc.size(0))
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)
            if f.startswith('uu'):
                sat[idx] = 0.0
            elif not parser.satisfied:
                sat[idx] = 0.0
            idx += 1
        num_literals = torch.tensor(num_literals)
        num_clauses = torch.tensor(num_clauses)
        torch.save([data_list, sat, num_literals, num_clauses],
                   osp.join(self.processed_dir,
                            self.processed_file_names))

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]  # , self.sat[idx]

