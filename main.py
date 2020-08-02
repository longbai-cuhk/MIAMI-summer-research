# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 10:56:02 2020

@author: gmy
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as Data
import numpy as np
import os
import resnet

BS = 40
LR = 0.001
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  
start_epoch = 0  

x = np.random.rand(1000,3*224*224+2)
y = np.random.randint(0,1,(1000,2))
x = torch.Tensor(x)
y = torch.Tensor(y)
y = torch.topk(y, 1)[1].squeeze(1)
print(y.size())
torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset=torch_dataset,      
    batch_size=BS,      
    shuffle=True,               
)
net0 = resnet.ResNet_ZERO50()
net0 = net0.to(device)
net = resnet.ResNet50()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR,
                      momentum=0.9, weight_decay=5e-4)

def train0(epoch):
    print('\nEpoch: %d' % epoch)
    net0.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        print(targets)
        targets = targets.long()
        inputs2 = inputs[:,0:2]
        
        inputs1 = inputs[:,2:]
        inputs1 = inputs1.view(40,3,224,224)
        optimizer.zero_grad()
        outputs = net0(inputs1,inputs2)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        print(targets)
        targets = targets.long()
        inputs2 = inputs[:,0:2]
        
        inputs1 = inputs[:,2:]
        inputs1 = inputs1.view(40,3,224,224)
        optimizer.zero_grad()
        outputs = net(inputs1,inputs2)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

for epoch in range(0,10):
    train(epoch)
