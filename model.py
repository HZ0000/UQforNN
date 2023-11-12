import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Net(nn.Module):
    def __init__(self, input_channel, hidden_layer):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_channel, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, 1)
        self.hidden_layer = hidden_layer


    def forward(self, x ):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = x * np.sqrt(2/self.hidden_layer)
        x = self.fc2(x)
        return x

def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        nn.init.normal_(m.weight.data, mean=0.0, std=1.0)
        nn.init.zeros_(m.bias)