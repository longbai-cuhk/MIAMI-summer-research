# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 11:15:17 2020

@author: lyonk
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as Data
import numpy as np


BS = 50
LR = 0.001
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  
start_epoch = 0  
#换成test_gender
x = np.load("test_age.npy")

print(x.shape)
x = (x-np.min(x)*np.ones(x.shape,dtype = np.float32))/(np.max(x)-np.min(x))

y = np.load("lab_test.npy")

print(y.shape)
y = y.astype(np.float32)
x0 = np.random.rand(50,10)
y0 = np.random.randint(0,2,(50,2))
y0 = y0.astype(np.float32)

x = torch.Tensor(x)
y = torch.Tensor(y)
x0 = torch.Tensor(x0)
y0 = torch.Tensor(y0)
#y = torch.topk(y, 1)[1].squeeze(1)
#print(y.size())
torch_dataset0 = Data.TensorDataset(x0, y0)
loader = Data.DataLoader(
    dataset=torch_dataset0,      
    batch_size=BS,      
    shuffle=False,               
)

torch_testset = Data.TensorDataset(x, y)
testloader = Data.DataLoader(
    dataset=torch_testset,      
    batch_size=BS,      
    shuffle=False,               
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
    def forward(self,x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.linear2(x)
        
        x = self.bn2(x)
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

net.load_state_dict(torch.load('net_epoch_withgaa2'+str(2)+'.pkl'))


def test():
    global best_acc,BS
    #net.eval()
    test_loss = 0
    correct0 = 0
    total = 0
    pt = 0
    pf = 0
    nt = 0
    nf = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            output = outputs.numpy()
            #print(output)
            for i in range(outputs.shape[0]):
                if outputs[i,0] > outputs[i,1]:
                    outputs[i,0] = 1.
                    outputs[i,1] = 0.
                else:
                    outputs[i,0] = 0.
                    outputs[i,1] = 1.
            outputs = outputs.view(targets.size())
            #print(outputs)
            #outputs = torch.Tensor(np.array([1., 0., 0., 1., 1., 1., 0., 0., 1., 1.]))
            #print(targets)
            total += targets.size(0)
            #correct = outputs.eq(targets).sum().item()
            correct = 0
            for i in range(targets.shape[0]):
                if outputs[i,0] == targets[i,0]:
                    correct += 1
                if outputs[i,0] == 1 and targets[i,0] == 1:
                    pt+=1
                elif outputs[i,0] == 0 and targets[i,0] == 1:
                    pf+=1
                elif outputs[i,0] == 1 and targets[i,0] == 0:
                    nf+=1
                else:
                    nt+=1
            correct0 += correct
            
    acc = 100.*correct0/total
    loss = BS*test_loss/total
    print('acc:')
    print(acc)
    print('loss:')
    print(loss)
    precision = pt/(pt+nf)
    recall = pt/(pt+pf)
    '''
    print(pt)
    print(pf)
    print(nt)
    print(nf)
    '''

    print("precision:")
    print(precision)
    print("recall:")
    print(recall)
def test0():
    global best_acc,BS
    #net.eval()
    test_loss = 0
    correct0 = 0
    total = 0
    pt = 0
    pf = 0
    nt = 0
    nf = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = net(inputs)
            
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            for i in range(outputs.shape[0]):
                if outputs[i,0] > outputs[i,1]:
                    outputs[i,0] = 1.
                    outputs[i,1] = 0.
                else:
                    outputs[i,0] = 0.
                    outputs[i,1] = 1.
            outputs = outputs.view(targets.size())
            #print(outputs)
            #outputs = torch.Tensor(np.array([1., 0., 0., 1., 1., 1., 0., 0., 1., 1.]))
            #print(targets)
            total += targets.size(0)
            #correct = outputs.eq(targets).sum().item()
            correct = 0
            for i in range(targets.shape[0]):
                if outputs[i,0] == targets[i,0]:
                    correct += 1
                if outputs[i,0] == 1 and targets[i,0] == 1:
                    pt+=1
                elif outputs[i,0] == 0 and targets[i,0] == 1:
                    pf+=1
                elif outputs[i,0] == 1 and targets[i,0] == 0:
                    nf+=1
                else:
                    nt+=1
            correct0 += correct
            
    acc = 100.*correct0/total
    loss = BS*test_loss/total
    print('train acc:')
    print(acc)
    print('train loss:')
    print(loss)
    precision = pt/(pt+nf)
    recall = pt/(pt+pf)
    '''
    print(pt)
    print(pf)
    print(nt)
    print(nf)
    '''

    print("precision:")
    
    print(precision)
    print("recall:")
    print(recall)
test()
test0()