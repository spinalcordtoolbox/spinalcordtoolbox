from torch import nn
import torch.nn.functional as F

class DownConv(nn.Module):
    def __init__(self, in_ch, out_ch, drop_rate=0.2, bn_momentum=0.1):
        super(DownConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1)
        self.conv_bn = nn.BatchNorm2d(out_ch, momentum=bn_momentum)
        self.conv_drop = nn.Dropout2d(drop_rate)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.conv_bn(x)
        x = self.conv_drop(x)
        return x

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Classifier(nn.Module):
    def __init__(self, drop_rate=0.2, bn_momentum=0.1):
        super(Classifier, self).__init__()
        self.conv1 = DownConv(1, 32, drop_rate, bn_momentum)
        self.mp1 = nn.MaxPool2d(2)
        
        self.conv2 = DownConv(32, 32, drop_rate, bn_momentum)
        self.mp2 = nn.MaxPool2d(2)
        
        self.conv3 = DownConv(32, 64, drop_rate, bn_momentum)
        self.mp3 = nn.MaxPool2d(2)       
        
        self.flat = Flatten()
        self.dense1 = nn.Linear(16384, 512)
        self.drop = nn.Dropout2d(drop_rate)
        self.dense2 = nn.Linear(512, 6)
        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.mp1(x1)
        
        x3 = self.conv2(x2)
        x4 = self.mp2(x3)
        
        x5 = self.conv3(x4)
        x6 = self.mp3(x5)
        
        x7 = self.flat(x6)
        x8 = F.relu(self.dense1(x7))
        x9 = self.drop(x8)
        x10 = self.dense2(x9)
        x11 = self.soft(x10)

        return(x11)
