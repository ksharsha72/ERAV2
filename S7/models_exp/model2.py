import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class SecondModel(nn.Module):
    def __init__(self):
        super(SecondModel, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, 3, padding=1),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout(0.1),
        )  # O_SIZE: 28, RF: 3
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 10, 3), nn.BatchNorm2d(10), nn.ReLU(), nn.Dropout(0.1)
        )  # O_SIZE: 26, RF: 5
        self.conv3 = nn.Sequential(
            nn.Conv2d(10, 20, 3), nn.BatchNorm2d(20), nn.ReLU(), nn.Dropout(0.1)
        )  # O_SIZE: 24, RF: 7
        self.trans1 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Sequential(
                nn.Conv2d(20, 10, (1, 1)),
                nn.BatchNorm2d(10),
                nn.ReLU(),
                nn.Dropout(0.1),
            ),
        )  # O_sIZE 12, RF:8
        self.conv4 = nn.Sequential(
            nn.Conv2d(10, 10, 3), nn.BatchNorm2d(10), nn.ReLU(), nn.Dropout(0.1)
        )  # O_SIZE: 10, RF: 12
        self.conv5 = nn.Sequential(
            nn.Conv2d(10, 10, 3), nn.BatchNorm2d(10), nn.ReLU(), nn.Dropout(0.1)
        )  # O_sIZE: 8, RF: 16
        self.conv6 = nn.Sequential(
            nn.Conv2d(10, 10, 3), nn.BatchNorm2d(10), nn.ReLU(), nn.Dropout(0.1)
        )  # O_SIZE: 6, RF: 20
        self.conv7 = nn.Sequential(
            nn.Conv2d(10, 18, 3, padding=1),
            nn.BatchNorm2d(18),
            nn.ReLU(),
            nn.Dropout(0.1),
        )  # O_SIZE: 6, RF:20
        self.gap = nn.Sequential(nn.AvgPool2d(6))
        self.conv8 = nn.Sequential(nn.Conv2d(18, 10, (1, 1)))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.trans1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.gap(x)
        x = self.conv8(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
