#!/home/pytorch/pytorch/sandbox/bin/python3

import torch
from torch import nn
from torch.nn import functionnal


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 3)
        self.split = self.split_tensor()
        self.conv2 = nn.Conv2d(12, 12, 2)

    def split_tensor(self):
        def func(tensor):
            return torch.cat(
                tensor[:, :2, :2], tensor[:, :2, 1:],
                tensor[:, 1:, :2], tensor[:, 1:, 1:])
        return func

    def forward(self, x):
        x = functionnal.max_pool2d(functionnal.relu(self.conv1(x)), (2, 2))
        x = self.split(x)
        x = functionnal.max_pool2d(functionnal.relu(self.conv2(x)), (2, 2))
        return x
