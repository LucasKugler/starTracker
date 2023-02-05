import torch
import torch.nn as nn
import torch.nn.functional as f

class StarTConv(nn.Module):
    def __init__(self, quarters=False):
        super(StarTConv, self).__init__()
        if quarters == True:
            outputDim = 4
        else:
            outputDim = 3
    
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(1,32,3)
        self.pool = nn.MaxPool2d(2,2)
        self.drop = nn.Dropout2d(0.5)
        self.conv2 = nn.Conv2d(32,64,3)
        self.conv3 = nn.Conv2d(64,128,3)
        self.fc1 = nn.Linear(128*13*8,128)
        self.fc2 = nn.Linear(128,outputDim)
        self.sm = nn.Softmax()

    def forward(self, x):
        out = self.drop(self.pool(f.relu(self.conv1(x))))
        out = self.drop(self.pool(f.relu(self.conv2(out))))
        out = self.drop(self.pool(f.relu(self.conv3(out))))
        out = out.view(-1, 128*13*8)
        out = self.drop(f.relu(self.fc1(out)))
        out = self.sm(self.fc2(out))
        
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