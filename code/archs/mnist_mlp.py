from __future__ import absolute_import

'''
Single layer MLP for MNIST, with test time noise (data and hidden layer)
'''
import torch
import torch.nn as nn
import math

import geotorch

# Fully connected neural network with one hidden layer
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, noise_std):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.LeakyReLU(negative_slope = 0.1) #leaky relu to keep invertible
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.input_size = input_size
        self.noise_std = noise_std
        geotorch.orthogonal(self.fc1, "weight") #first weight is orthogonal
    
    def forward(self, x):
        x = x.reshape(-1, self.input_size) #flatten layer
        if not self.training:
            x = x + self.noise_std[0] * torch.randn_like(x) #add noise to data
        out = self.fc1(x)
        out = self.relu(out)
        if not self.training:
            out = out + self.noise_std[1] * torch.randn_like(out) #add noise to hidden layer
        out = self.fc2(out)
        return out

def mlp(**kwargs):
    """
    Constructs a single hidden layer MLP model.
    """
    return MLP(**kwargs)