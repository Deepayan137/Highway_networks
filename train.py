import torch
import argparse

import numpy as np

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.autograd import Variable

from model import *
import pdb
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('filename', type=str)
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
help='input batch size for testing (default: 1000)')
parser.add_argument('--n_epochs', type=int, default=10, metavar='N',
help='number of epochs to train (default: 10)')
parser.add_argument('--learning_rate', type=float, default=0.01, metavar='LR',
help='learning rate (default: 0.01)')
parser.add_argument('--no-cuda', action='store_true', default=False,
help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
help='how many batches to wait before logging training status')
parser.add_argument('--highway-number', type=int, default=10, metavar='N',
help='how many highway layers to use in the model')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def save(model_ft):
    save_filename = args.filename
    torch.save(model_ft, save_filename)
    print('Saved as %s' % save_filename)


train_loader = torch.utils.data.DataLoader(
datasets.MNIST('../data', train=True, download=True,
                transform=transforms.Compose([transforms.ToTensor(),
                    ])),batch_size=args.batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
datasets.MNIST('../data', train=False, 
    transform=transforms.Compose([transforms.ToTensor(),
        ])),batch_size=args.batch_size, shuffle=True)
#transforms.Lambda(lambda x: x.numpy().flatten()),
batch_size = args.batch_size
def train_model(model_ft, criterion, optimizer, n_epochs):
    model_ft.train()
    
    for epoch in range(n_epochs):
        i=0
        correct = 0
        for batch, (data, target) in enumerate(train_loader):
            i+=1

            if args.cuda:
                data, target = data.cuda(), target.cuda()
            
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model_ft(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum()
            #pdb.set_trace()
        accuracy = (float(correct)/len(train_loader.dataset))*100
    
        print ('Epoch [%d/%d], Loss: %.4f, Accuracy: %.4f'%(epoch+1, n_epochs,  
         loss.data[0], accuracy))
        
    print("Saving..")
    save(model_ft)
    return model_ft

def test_model(model_ft, n_epochs):
    model_ft.eval()
    
    correct = 0
    for data, target in test_loader:
    
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)

        output = model_ft(data)

        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()
    accuracy = float(correct)/len(test_loader.dataset)*100
    print('Accuracy: %.4f'%(accuracy))

input_size = 784
output_size = 10
hidden_size = 512
model_ft = ConvNet()
if args.cuda:
    model_ft = model_ft.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_ft.parameters(), args.learning_rate)
try:
    train_model(model_ft, criterion, optimizer, args.n_epochs)
    #pass
except KeyboardInterrupt:
    print("Saving before quit...")
    save(model_ft)
model_ft=torch.load('conv.net')
test_model(model_ft, args.n_epochs)
