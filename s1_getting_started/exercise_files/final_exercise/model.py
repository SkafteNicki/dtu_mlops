from torch import nn
import torch

class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.convolution = nn.Sequential(
            nn.Conv2d(1,4,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(4,8,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
            #Size of image 7*7
        )
        #n_channels = self.convolution(torch.empty(1, 8, 7, 7)).size(-1)
        self.linear = nn.Sequential(
            nn.Linear(392,64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self,x):

        x = self.convolution(x)
        x = self.linear(x)

        return x



