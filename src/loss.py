from abc import ABC

import torch
import torch.nn as nn


class LabelSmoothing(nn.Module, ABC):
    def __init__(self):
        super(LabelSmoothing, self).__init__()


class SimpleLossCompute():
    def __init__(self):
        super(SimpleLossCompute, self).__init__()
