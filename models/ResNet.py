# -*- coding: utf-8 -*-
"""
@Author : Horizon
@Date   : 2020-12-31 01:12:20
"""

import torch
import torch.nn as nn

ResNet_models = ["ResNet18","ResNet34","ResNet50","ResNet101","ResNet152"]

def loadResNet(name="ResNet18", num_classes=1000, img_size=224):
    if(name == "ResNet18"):
        model = ResNet18(num_classes, img_size)
    elif(name == "ResNet34"):
        model = ResNet34(num_classes, img_size)
    elif(name == "ResNet50"):
        model = ResNet50(num_classes, img_size)
    elif(name == "ResNet101"):
        model = ResNet101(num_classes, img_size)
    elif(name == "ResNet152"):
        model = ResNet152(num_classes, img_size)
    else:
        model = None
    return model

class ResNet18(nn.Module):

    def __init__(self, num_classes=1000, img_size=224):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ResidualBlock(in_channels=64, hidden_channels=64, downsample=False),
            ResidualBlock(in_channels=64, hidden_channels=64, downsample=False),
        )

        self.conv3 = nn.Sequential(
            ResidualBlock(in_channels=64,  hidden_channels=128, downsample=True),
            ResidualBlock(in_channels=128, hidden_channels=128, downsample=False),
        )

        self.conv4 = nn.Sequential(
            ResidualBlock(in_channels=128, hidden_channels=256, downsample=True),
            ResidualBlock(in_channels=256, hidden_channels=256, downsample=False),
        )

        self.conv5 = nn.Sequential(
            ResidualBlock(in_channels=256, hidden_channels=512, downsample=True),
            ResidualBlock(in_channels=512, hidden_channels=512, downsample=False),
        )

        self.avpool = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Linear(in_features=512, out_features=num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.avpool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x

class ResNet34(nn.Module):

    def __init__(self, num_classes=1000, img_size=224):
        super(ResNet34, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ResidualBlock(in_channels=64, hidden_channels=64, downsample=False),
            ResidualBlock(in_channels=64, hidden_channels=64, downsample=False),
            ResidualBlock(in_channels=64, hidden_channels=64, downsample=False),
        )

        self.conv3 = nn.Sequential(
            ResidualBlock(in_channels=64,  hidden_channels=128, downsample=True),
            ResidualBlock(in_channels=128, hidden_channels=128, downsample=False),
            ResidualBlock(in_channels=128, hidden_channels=128, downsample=False),
            ResidualBlock(in_channels=128, hidden_channels=128, downsample=False),
        )

        self.conv4 = nn.Sequential(
            ResidualBlock(in_channels=128, hidden_channels=256, downsample=True),
            ResidualBlock(in_channels=256, hidden_channels=256, downsample=False),
            ResidualBlock(in_channels=256, hidden_channels=256, downsample=False),
            ResidualBlock(in_channels=256, hidden_channels=256, downsample=False),
            ResidualBlock(in_channels=256, hidden_channels=256, downsample=False),
            ResidualBlock(in_channels=256, hidden_channels=256, downsample=False),
        )

        self.conv5 = nn.Sequential(
            ResidualBlock(in_channels=256, hidden_channels=512, downsample=True),
            ResidualBlock(in_channels=512, hidden_channels=512, downsample=False),
            ResidualBlock(in_channels=512, hidden_channels=512, downsample=False),
        )

        self.avpool = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Linear(in_features=512, out_features=num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.avpool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x

class ResNet50(nn.Module):

    def __init__(self, num_classes=1000, img_size=224):
        super(ResNet50, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Bottleneck(in_channels=64,  down_channels=64, hidden_channels=64, up_channels=256, downsample=False),
            Bottleneck(in_channels=256, down_channels=64, hidden_channels=64, up_channels=256, downsample=False),
            Bottleneck(in_channels=256, down_channels=64, hidden_channels=64, up_channels=256, downsample=False),
        )

        self.conv3 = nn.Sequential(
            Bottleneck(in_channels=256, down_channels=128, hidden_channels=128, up_channels=512, downsample=True),
            Bottleneck(in_channels=512, down_channels=128, hidden_channels=128, up_channels=512, downsample=False),
            Bottleneck(in_channels=512, down_channels=128, hidden_channels=128, up_channels=512, downsample=False),
            Bottleneck(in_channels=512, down_channels=128, hidden_channels=128, up_channels=512, downsample=False),
        )

        self.conv4 = nn.Sequential(
            Bottleneck(in_channels=512,  down_channels=256, hidden_channels=256, up_channels=1024, downsample=True),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
        )

        self.conv5 = nn.Sequential(
            Bottleneck(in_channels=1024, down_channels=512, hidden_channels=512, up_channels=2048, downsample=True),
            Bottleneck(in_channels=2048, down_channels=512, hidden_channels=512, up_channels=2048, downsample=False),
            Bottleneck(in_channels=2048, down_channels=512, hidden_channels=512, up_channels=2048, downsample=False),
        )

        self.avpool = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Linear(in_features=2048, out_features=num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.avpool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x

class ResNet101(nn.Module):

    def __init__(self, num_classes=1000, img_size=224):
        super(ResNet101, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Bottleneck(in_channels=64,  down_channels=64, hidden_channels=64, up_channels=256, downsample=False),
            Bottleneck(in_channels=256, down_channels=64, hidden_channels=64, up_channels=256, downsample=False),
            Bottleneck(in_channels=256, down_channels=64, hidden_channels=64, up_channels=256, downsample=False),
        )

        self.conv3 = nn.Sequential(
            Bottleneck(in_channels=256, down_channels=128, hidden_channels=128, up_channels=512, downsample=True),
            Bottleneck(in_channels=512, down_channels=128, hidden_channels=128, up_channels=512, downsample=False),
            Bottleneck(in_channels=512, down_channels=128, hidden_channels=128, up_channels=512, downsample=False),
            Bottleneck(in_channels=512, down_channels=128, hidden_channels=128, up_channels=512, downsample=False),
        )

        self.conv4 = nn.Sequential(
            Bottleneck(in_channels=512,  down_channels=256, hidden_channels=256, up_channels=1024, downsample=True),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
        )

        self.conv5 = nn.Sequential(
            Bottleneck(in_channels=1024, down_channels=512, hidden_channels=512, up_channels=2048, downsample=True),
            Bottleneck(in_channels=2048, down_channels=512, hidden_channels=512, up_channels=2048, downsample=False),
            Bottleneck(in_channels=2048, down_channels=512, hidden_channels=512, up_channels=2048, downsample=False),
        )

        self.avpool = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Linear(in_features=2048, out_features=num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.avpool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x

class ResNet152(nn.Module):

    def __init__(self, num_classes=1000, img_size=224):
        super(ResNet152, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Bottleneck(in_channels=64,  down_channels=64, hidden_channels=64, up_channels=256, downsample=False),
            Bottleneck(in_channels=256, down_channels=64, hidden_channels=64, up_channels=256, downsample=False),
            Bottleneck(in_channels=256, down_channels=64, hidden_channels=64, up_channels=256, downsample=False),
        )

        self.conv3 = nn.Sequential(
            Bottleneck(in_channels=256, down_channels=128, hidden_channels=128, up_channels=512, downsample=True),
            Bottleneck(in_channels=512, down_channels=128, hidden_channels=128, up_channels=512, downsample=False),
            Bottleneck(in_channels=512, down_channels=128, hidden_channels=128, up_channels=512, downsample=False),
            Bottleneck(in_channels=512, down_channels=128, hidden_channels=128, up_channels=512, downsample=False),
            Bottleneck(in_channels=512, down_channels=128, hidden_channels=128, up_channels=512, downsample=False),
            Bottleneck(in_channels=512, down_channels=128, hidden_channels=128, up_channels=512, downsample=False),
            Bottleneck(in_channels=512, down_channels=128, hidden_channels=128, up_channels=512, downsample=False),
            Bottleneck(in_channels=512, down_channels=128, hidden_channels=128, up_channels=512, downsample=False),
        )

        self.conv4 = nn.Sequential(
            Bottleneck(in_channels=512,  down_channels=256, hidden_channels=256, up_channels=1024, downsample=True),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
            Bottleneck(in_channels=1024, down_channels=256, hidden_channels=256, up_channels=1024, downsample=False),
        )

        self.conv5 = nn.Sequential(
            Bottleneck(in_channels=1024, down_channels=512, hidden_channels=512, up_channels=2048, downsample=True),
            Bottleneck(in_channels=2048, down_channels=512, hidden_channels=512, up_channels=2048, downsample=False),
            Bottleneck(in_channels=2048, down_channels=512, hidden_channels=512, up_channels=2048, downsample=False),
        )

        self.avpool = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Linear(in_features=2048, out_features=num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.avpool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, hidden_channels, downsample):
        super(ResidualBlock, self).__init__()

        if downsample:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
            )

        self.weight_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=hidden_channels),
        )

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(num_features=hidden_channels),
            )
        else:
            self.downsample = None

        self.relu = nn.Sequential(
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):

        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x = x + identity

        x = self.relu(x)

        return x

class Bottleneck(nn.Module):

    def __init__(self, in_channels, down_channels, hidden_channels, up_channels, downsample):
        super(Bottleneck, self).__init__()

        if downsample:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=down_channels, kernel_size=1, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(num_features=down_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=down_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=down_channels),
                nn.ReLU(inplace=True),
            )

        self.weight_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=down_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.weight_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=up_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=up_channels),
        )

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=up_channels, kernel_size=1, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(num_features=up_channels),
            )
        elif (in_channels != up_channels):
            self.downsample = None
            self.up_scale = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=up_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=up_channels),
            )
        else:
            self.downsample = None
            self.up_scale = None

        self.relu = nn.Sequential(
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):

        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)
        x = self.weight_layer3(x)

        if self.downsample is not None:
            identity = self.downsample(identity)
        elif self.up_scale is not None:
            identity = self.up_scale(identity)

        x = x + identity

        x = self.relu(x)

        return x
