from __future__ import absolute_import

'''
MLP for MNIST, with test time noise (data and first hidden layer)
4 hidden layer is meant to have architecture
444 --> 200 --> 100 --> 100 --> 50 --> 10
'''
import torch
import torch.nn as nn
import math

import geotorch

# Fully connected neural network with one hidden layer
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, second_layer, num_classes, noise_std, nonlinear, long=0):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.LeakyReLU(negative_slope = 0.1) #leaky relu to keep invertible
        self.fc2 = nn.Linear(hidden_size, second_layer)
        self.fcshort = nn.Linear(second_layer, num_classes)
        self.fc3 = nn.Linear(second_layer, second_layer)
        self.fc4 = nn.Linear(second_layer, 50)
        self.fc5 = nn.Linear(50, num_classes)
        self.input_size = input_size
        self.noise_std = noise_std
        self.nonlinear = nonlinear
        self.long = long
#         geotorch.orthogonal(self.fc1, "weight") #first weight is orthogonal
    
    def forward(self, x):
#         x = x.reshape(-1, self.input_size) #flatten layer
#         if not self.training:
        x = x + self.noise_std[0] * torch.randn_like(x) #add noise to data
        h1 = self.fc1(x)
        if self.nonlinear:
            h1 = self.relu(h1)
#         if not self.training:
        nh1 = h1 + self.noise_std[1] * torch.randn_like(h1) #add noise to hidden layer
        h2 = self.fc2(nh1)
        h2 = self.relu(h2)
        if self.long:
            #3rd hidden layer
            h3 = self.fc3(h2)
            h3 = self.relu(h3)
            #4th hidden layer
            h4 = self.fc4(h3)
            h4 = self.relu(h4)
            out = self.fc5(h4)
        else:
            out = self.fcshort(h2)
        return out, h1

def mlp(**kwargs):
    """
    Constructs a single hidden layer MLP model.
    """
    return MLP(**kwargs)