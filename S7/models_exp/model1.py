import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class FirstModel(nn.Module):
    def __init__(self):
        super(FirstModel,self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(1,10,3,padding=1),nn.ReLU()) #OUT_SIZE: 28 , RF 3
        self.conv2 = nn.Sequential(nn.Conv2d(10,10,3),nn.ReLU()) #OUT_SIZE: 26, RF 5
        self.conv3 = nn.Sequential(nn.Conv2d(10,10,3),nn.ReLU()) #OUT_SIZE: 24, RF 7
        self.pool1 = nn.MaxPool2d(2,2) #OUT_SIZE:12, RF: 8
        self.trans1 = nn.Sequential(nn.Conv2d(10,8,(1,1)),nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(8,10,3),nn.ReLU())#OUT_SIZE: 10, RF 12
        self.conv5 = nn.Sequential(nn.Conv2d(10,8,3),nn.ReLU())#OUT_sIZE: 8, RF:16
        self.conv6 = nn.Sequential(nn.Conv2d(8,10,3),nn.ReLU())#OUT_sIZE: 6, RF: 20
        self.conv7 = nn.Sequential(nn.Conv2d(10,10,6))
    
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
        x = x.view(-1,10)
        return F.log_softmax(x,dim=-1)
1