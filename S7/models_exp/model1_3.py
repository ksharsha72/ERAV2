import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ModelV3(nn.Module):
    def __init__(self) -> None:
        super(ModelV3,self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(1,10,3,padding=1),nn.BatchNorm2d(10),nn.ReLU(),nn.Dropout(0.1)) #output: 28, RF: 3
        self.conv2 = nn.Sequential(nn.Conv2d(10,12,3),nn.BatchNorm2d(12),nn.ReLU(),nn.Dropout(0.1))#output: 26, RF: 5
        self.trans1 = nn.Sequential(nn.Conv2d(12,10,(1,1)),nn.BatchNorm2d(10),nn.ReLU(),nn.Dropout(0.1))#output: 26, RF: 5
        self.pool1 = nn.Sequential(nn.MaxPool2d(2,2))#output: 13, RF: 6
        self.conv3 = nn.Sequential(nn.Conv2d(10,6,3),nn.BatchNorm2d(6),nn.ReLU(),nn.Dropout(0.1))#output : 11, RF: 10
        self.conv4 = nn.Sequential(nn.Conv2d(6,10,3),nn.BatchNorm2d(10),nn.ReLU(),nn.Dropout(0.1))#output: 9, RF: 14
        self.conv5 = nn.Sequential(nn.Conv2d(10,12,3),nn.BatchNorm2d(12),nn.ReLU(),nn.Dropout(0.1))#output: 7, RF: 18
        self.conv6 = nn.Sequential(nn.Conv2d(12,16,3,padding=1),nn.BatchNorm2d(16),nn.ReLU(),nn.Dropout(0.1))#output: 7, RF: 22
        self.conv7 = nn.Sequential(nn.Conv2d(16,20,3,padding=1),nn.BatchNorm2d(20),nn.ReLU(),nn.Dropout(0.1))#output: 5, RF: 26
        self.conv8 = nn.AvgPool2d(5)
        self.conv9 = nn.Conv2d(20,10,1)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.trans1(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = x.view(-1,10)
        return F.log_softmax(x,dim=-1) 