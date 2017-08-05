import math
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import pdb



class Highway(nn.Module):
	def __init__(self, input_size, activation, carry_bias = -1):
		super(Highway, self).__init__()
        #y = H*T + x*(1 - T)
		self.input_size = input_size
		self.activation = activation
		self.weight_T = Parameter(torch.Tensor(self.input_size, self.input_size))
		self.bias_T = Parameter(torch.Tensor(self.input_size*[carry_bias]))
		self.weight = Parameter(torch.Tensor(self.input_size, self.input_size))
		self.bias = Parameter(torch.Tensor(self.input_size*[-0.1]))

	def forward(self, x):
		
		T = torch.sigmoid(x.mm(self.weight_T)+self.bias_T.view(1,-1).expand_as(x.mm(self.weight_T)))
		H = self.activation(x.mm(self.weight)+self.bias.view(1, -1).expand_as(x.mm(self.weight)))
		C = (1.0 - T)

		output = torch.add(torch.mul(H, T), torch.mul(x, C))
		return output


