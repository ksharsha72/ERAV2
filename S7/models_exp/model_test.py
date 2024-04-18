import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Model1(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Model1,self).__init__(*args, **kwargs)

        self.conv1 = nn.Conv2d(1,10,3,padding=1)
        self.conv2 = nn.Conv2d(10,10,3,padding=1)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(10,10,3)
        self.conv4 = nn.Conv2d(10,10,3)
        self.conv5 = nn.Conv2d(10,10,3,padding=1)
        self.pool2 = nn.MaxPool2d(2,2)
        self.conv6 = nn.Conv2d(10,20,3)
        self.conv7 = nn.Conv2d(20,10,3)
    
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool2(x)
        x = F.relu(self.conv6(x))
        x = self.conv7(x)
        x = x.view(-1,10)
        return F.log_softmax(x,dim=-1)


