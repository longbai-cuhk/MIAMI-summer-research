
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as Data
import numpy as np
import os
import densenet
import densenet_agegender
import resnet
import resnet_agegender

BS = 50
LR = 0.001
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  
start_epoch = 0  

"""
x = np.load("trainenhanced_withgaa_1234_random.npy")
print(x.shape)
y = np.load("lab_train_withgaa_1234_random.npy")
print(y.shape)
"""
x = np.load("trainenhanced_withoutgaa_1234_random.npy")
print(x.shape)
y = np.load("lab_train_normal_1234_random.npy")
print(y.shape)

x = torch.Tensor(x)
y = torch.Tensor(y)
#y = torch.topk(y, 1)[1].squeeze(1)
#print(y.size())
torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset=torch_dataset,      
    batch_size=BS,      
    shuffle=True,               
)


net = densenet.densenet121(in_channel=6)
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
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        #print(targets)
        targets = targets.long()
        optimizer.zero_grad()
        outputs = net(inputs)
        targets = targets.float()
        
        loss = criterion(outputs,targets)
        print(loss)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        #total += targets.size(0)
        #correct += predicted.eq(targets).sum().item()

#net.load_state_dict(torch.load("GAAresnet50_epoch8.pkl"))

for epoch in range(0,50):
    train(epoch)
    """
    if epoch > 3 :
      LR = 0.001
      if epoch > 5 :
        LR = 0.0002
    """
    torch.save(net.state_dict(), '0.001densenet121_epoch'+str(epoch)+'.pkl')