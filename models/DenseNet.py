# -*- coding: utf-8 -*-
"""
@Author : Horizon
@Date   : 2021-01-14 10:44:20
"""

import torch
import torch.nn as nn

DenseNet_models = ["DenseNet121","DenseNet169","DenseNet201","DenseNet264"]

def loadDenseNet(name="ResNet18", num_classes=1000, img_size=224):
    if(name == "DenseNet121"):
        model = DenseNet121(num_classes, img_size)
    elif(name == "DenseNet169"):
        model = DenseNet169(num_classes, img_size)
    elif(name == "DenseNet201"):
        model = DenseNet201(num_classes, img_size)
    elif(name == "DenseNet264"):
        model = DenseNet264(num_classes, img_size)
    else:
        model = None
    return model

def DenseNet121(num_classes=1000, img_size=224, k=32):

    return DenseNet(num_classes=num_classes, img_size=img_size, k=k, config=[6, 12, 24, 16])

def DenseNet169(num_classes=1000, img_size=224, k=32):

    return DenseNet(num_classes=num_classes, img_size=img_size, k=k, config=[6, 12, 32, 32])

def DenseNet201(num_classes=1000, img_size=224, k=32):

    return DenseNet(num_classes=num_classes, img_size=img_size, k=k, config=[6, 12, 48, 32])

def DenseNet264(num_classes=1000, img_size=224, k=32):

    return DenseNet(num_classes=num_classes, img_size=img_size, k=k, config=[6, 12, 64, 48])

class DenseNet(nn.Module):

    def __init__(self, num_classes=1000, img_size=224, k=32, config=[6, 12, 24, 16]):
        super(DenseNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=2*k, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=2*k),
            nn.ReLU(inplace=True),
        )

        num_features = 2*k

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block1 = DenseBlock(input_features=2*k, layers=config[0], k=k)

        num_features += config[0]*k

        self.trans1 = Transition(input_features=num_features, output_features=num_features//2)

        num_features = num_features // 2

        self.block2 = DenseBlock(input_features=num_features, layers=config[1], k=k)

        num_features += config[1]*k

        self.trans2 = Transition(input_features=num_features, output_features=num_features//2)

        num_features = num_features // 2

        self.block3 = DenseBlock(input_features=num_features, layers=config[2], k=k)

        num_features += config[2]*k

        self.trans3 = Transition(input_features=num_features, output_features=num_features//2)

        num_features = num_features // 2

        self.block4 = DenseBlock(input_features=num_features, layers=config[3], k=k)

        num_features += config[3]*k

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Linear(in_features=num_features, out_features=num_classes)
    
    def forward(self, x):

        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.block1(x)
        x = self.trans1(x)

        x = self.block2(x)
        x = self.trans2(x)

        x = self.block3(x)
        x = self.trans3(x)

        x = self.block4(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x

class DenseLayer(nn.Module):

    def __init__(self, input_features, k):
        super(DenseLayer, self).__init__()

        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(num_features=input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=input_features, out_channels=4*k, kernel_size=1, bias=False),
        )

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(num_features=4*k),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=4*k, out_channels=k, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, x):

        x = self.conv1(x)

        x = self.conv2(x)

        return x

class DenseBlock(nn.Module):

    def __init__(self, input_features, layers, k):
        super(DenseBlock, self).__init__()

        self.blocks = []

        for i in range(layers):

            #self.blocks.append(DenseLayer(input_features=input_features+i*k, k=k))

            layer = DenseLayer(input_features=input_features+i*k, k=k)

            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, x):

        for name, layer in self.named_children():

            new_x = layer(x)

            x = torch.cat((x, new_x), 1)

        return x

class Transition(nn.Module):

    def __init__(self, input_features, output_features):
        super(Transition, self).__init__()

        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(num_features=input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=input_features, out_channels=output_features, kernel_size=1, bias=False),
        )

        self.avpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):

        x = self.conv1(x)

        x = self.avpool(x)

        return x