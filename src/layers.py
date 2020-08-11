import torch.nn as nn
import copy


def clones(module, k):
    return nn.ModuleList(
        copy.deepcopy(module) for _ in range(k)
    )

