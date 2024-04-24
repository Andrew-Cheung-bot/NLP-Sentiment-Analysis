import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, output_size):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(768, 1024)
        self.layer2 = nn.Linear(1024, output_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.layer1(x))
        x = self.tanh(self.layer2(x))
        return x

