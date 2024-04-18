import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ModelV2(nn.Module):
    def __init__(self) -> None:
        super(ModelV2,self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(1,10,3),nn.BatchNorm2d(10),nn.ReLU()) #output: 26, RF: 3
        self.conv2 = nn.Sequential(nn.Conv2d(10,8,3),nn.BatchNorm2d(8),nn.ReLU())#output: 24, RF: 5
        self.conv3 = nn.Sequential(nn.Conv2d(8,14,3),nn.BatchNorm2d(14),nn.ReLU())#output: 22, RF: 7
        self.pool1 = nn.Sequential(nn.MaxPool2d(2,2))#output: 11, RF: 8
        self.trans1 = nn.Sequential(nn.Conv2d(14,12,(1,1)),nn.BatchNorm2d(12),nn.ReLU())#output: 11, RF: 8
        self.conv4 = nn.Sequential(nn.Conv2d(12,8,3),nn.BatchNorm2d(8),nn.ReLU())#output : 9, RF: 12
        self.conv5 = nn.Sequential(nn.Conv2d(8,12,3),nn.BatchNorm2d(12),nn.ReLU())#output: 7, RF: 16
        self.conv6 = nn.Sequential(nn.Conv2d(12,10,3),nn.BatchNorm2d(10),nn.ReLU())#output: 5, RF: 20
        self.conv7 = nn.Sequential(nn.Conv2d(10,12,3),nn.BatchNorm2d(12),nn.ReLU())#output: 3, RF: 24
        self.conv8 = nn.Sequential(nn.Conv2d(12,10,(1,1)),nn.BatchNorm2d(10),nn.ReLU())#output: 3, RF: 24
        self.conv9 = nn.Sequential(nn.Conv2d(10,10,3))#output 1, RF: 28

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool1(x)
        x = self.trans1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = x.view(-1,10)
        return F.log_softmax(x,dim=-1) 