# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 19:17:15 2020

@author: lyonk
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 15:14:43 2020

@author: lyonk
"""

import torch
import torch.nn as nn
import torchvision
import numpy as np

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

__all__ = ['ResNet50', 'ResNet101','ResNet152']

def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=7,stride=stride,padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

class Bottleneck(nn.Module):
    def __init__(self,in_places,places, stride=1,downsampling=False, expansion = 4):
        super(Bottleneck,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places*self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self,blocks, num_classes=2, expansion = 4, is_double = 0):
        super(ResNet,self).__init__()
        self.expansion = expansion
        if is_double == 0:
            a = 3
        else:
            a = 6
        self.conv1 = Conv1(in_planes = a, places= 64)

        self.layer1 = self.make_layer(in_places = 64, places= 64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places = 256,places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512,places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024,places=512, block=blocks[3], stride=2)
        self.is_double = is_double
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(2048,6)
        self.fc2 = nn.Linear(12,num_classes)
        self.bn = nn.BatchNorm1d(6)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(2)
        self.ln1 = nn.Linear(2,6)
        self.bn3 = nn.BatchNorm1d(6)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places,stride, downsampling =True))
        for i in range(1, block):
            layers.append(Bottleneck(places*self.expansion, places))

        return nn.Sequential(*layers)


    def forward(self, x):
        x1 = x[:,0:2]
        x = x[:,2:]
        if self.is_double ==0:    
            x = x.view(-1,3,224,224)
        else:
            x = x.view(-1,6,224,224)
        #print(x.size())
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        x1 = self.ln1(x1)
        x1 = self.bn3(x1)
        x1 = self.relu(x1)
        x = torch.cat((x,x1),1)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.tanh(x)
        return x
    

    
def ResNet50(db = 0):
    return ResNet([3, 4, 6, 3],is_double = db)

def ResNet101(db = 0):
    return ResNet([3, 4, 23, 3],is_double = db)

def ResNet152(db = 0):
    return ResNet([3, 8, 36, 3],is_double = db)
    
    
if __name__=='__main__':
    model = ResNet50(1)
    #print(model)

    input = torch.randn(2, 6*224*224+2)
    out = model(input)
    #print(out.shape)
