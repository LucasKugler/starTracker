import torch
import torch.nn as nn
import torch.nn.functional as f

class StarTConv(nn.Module):
    def __init__(self):
        super(StarTConv, self).__init__()
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(1,3,5,2)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(3,8,5)
        self.fc1 = nn.Linear(8*12*7,120)
        self.fc2 = nn.Linear(120,50)
        self.fc3 = nn.Linear(50,3)

    def forward(self, x):
        out = self.pool(f.relu(self.conv1(x)))
        out = self.pool(f.relu(self.conv2(x)))
        out = out.view(-1, 8*12*7)
        out = f.relu(self.fc1(out))
        out = f.relu(self.fc2(out))
        out = self.fc3(out)
        
        return out

class StarTModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(StarTModel, self).__init__()
        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x = self.flatten(x)
        out = self.lin1(x)
        out = self.relu(out)
        out = self.lin2(out)
        # out = self.relu(out)
        out = self.lin3(out)
        
        return out