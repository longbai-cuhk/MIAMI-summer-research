# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 11:06:12 2020

@author: lyonk
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as Data
import numpy as np

BS = 50
LR = 0.01
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  
start_epoch = 0  

#x = np.load("train2gaa.npy")
#x = (x-np.min(x)*np.ones(x.shape,dtype = np.float32))/(np.max(x)-np.min(x))
#y = np.load("ssssecondtestlabel_withgaa_random.npy")
#换成train2_gender
x = np.load("train2_age.npy")
y = np.load("lab_train2_12_random.npy")
y = y.astype(np.float32)
print(y.dtype)
x = torch.Tensor(x)
y = torch.Tensor(y)
#y = torch.topk(y, 1)[1].squeeze(1)
print(y.size())
torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset=torch_dataset,      
    batch_size=BS,      
    shuffle=True,               
)
class linear(nn.Module):
    def __init__(self):
        super(linear,self).__init__()
        self.linear1 = nn.Linear(10,240)
        self.linear2 = nn.Linear(240,120)
        self.linear3 = nn.Linear(120,2)
        self.bn1 = nn.BatchNorm1d(240)
        self.bn2 = nn.BatchNorm1d(120)
        self.bn3 = nn.BatchNorm1d(2)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
    def forward(self,x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.bn3(x)
        x = self.tanh(x)
        return x
net = linear()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=LR,
                      momentum=0.9, weight_decay=5e-4)



def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    loss0 = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        targets = targets.long()
        optimizer.zero_grad()
        outputs = net(inputs)
        targets = targets.float()
        
        loss = criterion(outputs,targets)
        loss0+=loss
        #print(loss)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        #total += targets.size(0)
        #correct += predicted.eq(targets).sum().item()
    print(loss0)
for epoch in range(0,20):
    '''
    if epoch >50:
        LR = LR/10
    elif epoch>100:
        LR = LR/5
    elif epoch>150:
        LR = LR/4
    '''
    train(epoch)
    torch.save(net.state_dict(), 'net_epoch_withgaa2'+str(epoch)+'.pkl')
