import torch
import torch.nn as nn

class StarTModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(StarTModel, self).__init__()
        self.flatten = nn.Flatten()
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.flatten(x)
        return self.lin(x)