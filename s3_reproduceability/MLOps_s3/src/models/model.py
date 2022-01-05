import torch.nn.functional as F
from torch import nn, optim
from torch.nn import BatchNorm2d, Conv2d, Dropout2d, Linear, MaxPool2d
from torch.nn.functional import elu, relu, relu6, sigmoid, softmax, tanh


class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        
    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)
        
        return x


   


# define network
class MyAwesomeConvolutionalModel(nn.Module):
    def __init__(self):#, output_size):
        super().__init__()
        self.num_classes = 10#output_size
        
        # First convolutional layer
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding = (1, 1))
        self.batchnorm1 = nn.BatchNorm2d(10)
        # Second convolutional layer + pooling
        self.conv_2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size = 3, stride=1, padding = (1, 1))
        self.batchnorm2 = nn.BatchNorm2d(20)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 28 -> 14
        # Third convolutional layer
        self.conv_3 = nn.Conv2d(in_channels=20, out_channels=30, kernel_size = 3, stride=1, padding = (1, 1))
        self.batchnorm3 = nn.BatchNorm2d(30)
        # Thourth convolutional layer + pooling
        self.conv_4 = nn.Conv2d(in_channels=30, out_channels=30, kernel_size = 3, stride=1, padding = (1, 1))
        self.batchnorm4 = nn.BatchNorm2d(30)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # 14 -> 7
        # Ouput layer
        self.in_features = 7*7*30
        self.fc1 = nn.Linear(in_features=self.in_features, 
                          out_features=300,
                          bias=True)
        self.l_out = nn.Linear(in_features=300, 
                            out_features=self.num_classes,
                            bias=False)
        
    def forward(self, x):
       
        #x = x.permute(0, 3, 1, 2)
        x = F.relu(self.batchnorm1(self.conv_1(x)))
        #x = self.dropout(x)
        x = F.relu(self.batchnorm2(self.conv_2(x)))
        x = self.pool2(x)
        x = F.relu(self.batchnorm3(self.conv_3(x)))
        #x = self.dropout(x)
        x = F.relu(self.batchnorm4(self.conv_4(x)))
        x = self.pool4(x)
        x = x.view(x.size(0), -1)
        #print(x.size())
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.l_out(x), dim = 1)
        pip install black
        return x