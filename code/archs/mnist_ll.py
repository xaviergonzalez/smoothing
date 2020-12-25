from __future__ import absolute_import

'''
MLP for MNIST, with test and train time noise (data and first hidden layer)
4 hidden layer is meant to have architecture
444 --> 200 --> 20 --> 10
Either ALL linear, or ALL nonlinear (leaky ReLU)
pert allows for the addition of a random perturbation with l_2 norm pert
'''
import torch
import torch.nn as nn
import math

class MLP(nn.Module):
    def __init__(self, noise_std, nonlinear, pert=None):
        super(MLP, self).__init__()
        d0 = 444 #input size
        d1 = 200 #width of first hidden layer
        d2 = 20 #width of second hidden layer
        num_classes = 10 #10 classes bc MNIST
        self.fc1 = nn.Linear(d0, d1, bias = False) 
        self.relu = nn.LeakyReLU(negative_slope = 0.1) #leaky relu to keep invertible
        self.fc2 = nn.Linear(d1, d2, bias = False)
        self.fc3 = nn.Linear(d2, num_classes)
        self.noise_std = noise_std
        self.nonlinear = nonlinear
        self.d2 = d2
        self.pert = pert
    
    def forward(self, x):
        x = x + self.noise_std[0] * torch.randn_like(x) #add noise to data
        h1 = self.fc1(x)
        if self.nonlinear:
            h1 = self.relu(h1)
        nh1 = h1 + self.noise_std[1] * torch.randn_like(h1) #add noise to first hidden layer
        h2 = self.fc2(nh1)
        if self.nonlinear:
            h2 = self.relu(h2)
        if self.pert != None: #if we perturb the second hidden layer
            h2 = h2 + self.pert
        nh2 = h2 + self.noise_std[2] * torch.randn_like(h2) #add noise to second hidden layer
        out = self.fc3(nh2)
        return out, h1, h2

def mlp(**kwargs):
    """
    Constructs a single hidden layer MLP model.
    """
    return MLP(**kwargs)