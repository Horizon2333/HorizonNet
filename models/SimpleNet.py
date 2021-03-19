import torch
import torch.nn as nn

class SimpleNet96(nn.Module):

    def __init__(self, num_classes=1000, img_size=96):
        super(SimpleNet96, self).__init__()

        self.img_size = img_size

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)           
        )

        self.img_size = self.img_size / 2

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)            
        )

        self.img_size = self.img_size / 2

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            #nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(num_features=256),
            #nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)              
        )

        self.img_size = self.img_size / 2

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            #nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(num_features=512),
            #nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)              
        )

        self.img_size = self.img_size / 2

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=num_classes)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)

    
    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x

