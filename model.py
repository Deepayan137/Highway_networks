from layer import Highway
import torch
import argparse

import numpy as np

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.autograd import Variable


class Model(nn.Module):
	def __init__(self, input_size, output_size):
		super(Model, self).__init__()
		self.highway_layers = nn.ModuleList([Highway(input_size, activation=F.relu) for _ in range(10)])
		self.linear  = nn.Linear(input_size, output_size)

	def forward(self, x):
		for layer in self.highway_layers:
			x = layer(x)
		output = F.softmax(self.linear(x))
		return output


class ConvNet(nn.Module):
	def __init__(self):
		super(ConvNet, self).__init__()

		self.layer1 = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=5, padding=2),
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.MaxPool2d(2))
		self.layer2 = nn.Sequential(
			nn.Conv2d(16, 32, kernel_size=5, padding=2),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d(2))
		self.fc   = nn.Linear(7*7*32, 10)
		


	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = out.view(out.size(0), -1)
		out = self.fc_in(out)
		

		return out

class MLP(nn.Module):
	def __init__(self, input_size, output_size, hidden_size):
		super(MLP, self).__init__()
		self.fc_in = nn.Linear(input_size, hidden_size)
		self.fc_hidden= nn.Linear(hidden_size, 256)
		self.fc_out = nn.Linear(256, output_size)

	def forward(self, x):
		x = self.fc_in(x)
		x = self.fc_hidden(x)
		out = self.fc_out(x)
		return out
