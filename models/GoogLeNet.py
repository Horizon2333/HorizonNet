import torch
import torch.nn as nn

GoogLeNetmodels = ["GoogLeNet","GoogLeNet_aux"]

def loadGoogLeNet(name="GoogLeNet", num_classes=1000, img_size=224):
    if(name == "GoogLeNet"):
        model = GoogLeNet(num_classes, img_size)
    elif(name == "GoogLeNet_aux"):
        model = GoogLeNet_aux(num_classes, img_size)
    else:
        model = None
    return model

class GoogLeNet(nn.Module):

    def __init__(self, num_classes=1000, img_size=224):
        super(GoogLeNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True)
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.lrn = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception3a = inception(192,  64,  96, 128, 16, 32, 32)
        self.inception3b = inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = inception(480, 192,  96, 208, 16,  48,  64)
        self.inception4b = inception(512, 160, 112, 224, 24,  64,  64)
        self.inception4c = inception(512, 128, 128, 256, 24,  64,  64)
        self.inception4d = inception(512, 112, 144, 288, 32,  64,  64)
        self.inception4e = inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 

        self.inception5a = inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features=1024, out_features=num_classes)
        )
    
    def forward(self, x):

        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.lrn(x)

        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x

class GoogLeNet_aux(nn.Module):

    def __init__(self, num_classes=1000, img_size=224):
        super(GoogLeNet_aux, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True)
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.lrn = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception3a = inception(192,  64,  96, 128, 16, 32, 32)
        self.inception3b = inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = inception(480, 192,  96, 208, 16,  48,  64)
        self.inception4b = inception(512, 160, 112, 224, 24,  64,  64)
        self.inception4c = inception(512, 128, 128, 256, 24,  64,  64)
        self.inception4d = inception(512, 112, 144, 288, 32,  64,  64)
        self.inception4e = inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.aux1 = AuxiliaryClassifier(512, num_classes)
        self.aux2 = AuxiliaryClassifier(528, num_classes)

        self.inception5a = inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features=1024, out_features=num_classes)
        )
    
    def forward(self, x):

        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.lrn(x)

        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)

        aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        aux2 = self.aux2(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)
        if self.training:
            return x, aux1, aux2
        else:
            return x

class inception(nn.Module):

    def __init__(self, in_channels, conv1x1_channels, conv3x3_reduce_channels, conv3x3_channels, 
                conv5x5_reduce_channels, conv5x5_channels, pool_proj):
        super(inception, self).__init__()

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=conv1x1_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=conv3x3_reduce_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=conv3x3_reduce_channels, out_channels=conv3x3_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.conv5x5 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=conv5x5_reduce_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=conv5x5_reduce_channels, out_channels=conv5x5_channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )

        self.maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=in_channels, out_channels=pool_proj, kernel_size=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):

        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x3 = self.conv5x5(x)
        x4 = self.maxpool(x)

        #print(x1.shape, x2.shape, x3.shape, x4.shape)

        x = torch.cat((x1, x2, x3, x4), 1)
        #x = torch.cat((self.conv1x1(x), self.conv3x3(x), self.conv5x5(x), self.maxpool(x)), 1)

        return x

class AuxiliaryClassifier(nn.Module):
    def __init__(self, in_channels, num_classes=1000):
        super(AuxiliaryClassifier, self).__init__()

        self.avpool = nn.AdaptiveAvgPool2d((4,4))

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=4*4*128, out_features=1024),
            nn.Dropout(p=0.7),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=num_classes)
        )
    
    def forward(self, x):

        x = self.avpool(x)
        x = self.conv1x1(x)
        
        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x