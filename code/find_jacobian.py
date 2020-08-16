"""
Finds the jacobian of the output wrt the inputs 
adapted from: https://github.com/ast0414/adversarial-example/blob/master/craft.py
Need to better understand this function...why is it scaling worse over time...
are the gradients actually getting more complicated, or is this some feature of the function itself?
"""


import numpy as np

from seaborn import heatmap

import torch
import torch.nn as nn
import math
from torchvision import datasets, transforms

from torch.autograd.gradcheck import zero_gradients

import geotorch

def find_jacobian(inputs, output):
	"""
	:param inputs: Batch X Size (e.g. Depth X Width X Height)
	:param output: Batch X Classes
	:return: jacobian: Batch X Classes X Size
	"""
	assert inputs.requires_grad

	num_classes = output.size()[0]

	jacobian = torch.zeros(num_classes, *inputs.size())
	grad_output = torch.zeros(*output.size())
	if inputs.is_cuda:
		grad_output = grad_output.cuda()
		jacobian = jacobian.cuda()

	for i in range(num_classes):
		zero_gradients(inputs)
		grad_output.zero_()
		grad_output[i] = 1
		output.backward(grad_output, retain_graph=True)
		jacobian[i] = inputs.grad.data

	return jacobian