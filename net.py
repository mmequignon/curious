#!/home/pytorch/pytorch/sandbox/bin/python3

from torch import nn, optim
from torch.nn import functional


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 36, 1)
        self.conv2 = nn.Conv2d(36, 144, 2)
        self.conv3 = nn.Conv2d(144, 36, 2)
        self.linear = nn.Linear(36, 9)
        self.softmax = nn.Softmax(1)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.7)

    def forward(self, x):
        x = functional.relu(self.conv1(x))
        x = functional.relu(self.conv2(x))
        x = functional.relu(self.conv3(x))
        x = x.view(-1, 36)
        x = functional.relu(self.linear(x))
        x = self.softmax(x)
        return x
