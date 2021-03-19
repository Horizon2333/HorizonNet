import torch
import torch.nn as nn

AlexNet_models = ["AlexNet", "AlexNetv1"]

def loadAlexNet(name="VGG19", num_classes=1000, img_size=224):
    if(name == "AlexNet"):
        model = AlexNet(num_classes, img_size)
    elif(name == "AlexNetv1"):
        model = AlexNetv1(num_classes, img_size)
    else:
        model = None
    return model

class AlexNet(nn.Module):

    def __init__(self, num_classes=1000, img_size=224):

        super(AlexNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.ReLU(inplace=True)                   
        )

        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.ReLU(inplace=True)          
        )

        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.ReLU(inplace=True),            
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.ReLU(inplace=True),              
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.ReLU(inplace=True),              
        )

        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fc = nn.Sequential(
            nn.Linear(in_features=int(6*6*256), out_features=4096),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=num_classes)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 1)
        
        nn.init.constant_(self.conv1[0].bias, 0)
        nn.init.constant_(self.conv3[0].bias, 0)
    
    def forward(self, x):

        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x

class AlexNetv1(nn.Module):

    def __init__(self, num_classes=1000, img_size=224):

        super(AlexNetv1, self).__init__()

        self.branch1_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=11, stride=4, padding=2),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.ReLU(inplace=True)                   
        )

        self.branch1_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.branch1_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.ReLU(inplace=True)          
        )

        self.branch1_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.branch1_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.ReLU(inplace=True),            
        )

        self.branch1_conv4 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.ReLU(inplace=True),              
        )

        self.branch1_conv5 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.ReLU(inplace=True),              
        )

        self.branch1_pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.branch1_fc1 = nn.Sequential(
            nn.Linear(in_features=int(6*6*256), out_features=2048),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True)          
        )

        self.branch1_fc2 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=2048),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True)            
        )

        self.branch2_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=11, stride=4, padding=2),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.ReLU(inplace=True)                   
        )

        self.branch2_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.branch2_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.ReLU(inplace=True)          
        )

        self.branch2_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.branch2_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.ReLU(inplace=True),            
        )

        self.branch2_conv4 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.ReLU(inplace=True),              
        )

        self.branch2_conv5 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.ReLU(inplace=True),              
        )

        self.branch2_pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.branch2_fc1 = nn.Sequential(
            nn.Linear(in_features=int(6*6*256), out_features=2048),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True)          
        )

        self.branch2_fc2 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=2048),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True)            
        )

        self.fc = nn.Linear(in_features=4096, out_features=num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 1)
        
        nn.init.constant_(self.branch1_conv1[0].bias, 0)
        nn.init.constant_(self.branch1_conv3[0].bias, 0)

        nn.init.constant_(self.branch2_conv1[0].bias, 0)
        nn.init.constant_(self.branch2_conv3[0].bias, 0)
    
    def forward(self, x):

        #print('x    ',x.shape)

        x1 = self.branch1_conv1(x)
        x2 = self.branch2_conv1(x)

        #print('conv1', x1.shape, x2.shape)

        x1 = self.branch1_pool1(x1)
        x2 = self.branch2_pool1(x2)

        #print('pool1', x1.shape, x2.shape)

        x1 = self.branch1_conv2(x1)
        x2 = self.branch2_conv2(x2)

        #print('conv2', x1.shape, x2.shape)

        x1 = self.branch1_pool2(x1)
        x2 = self.branch2_pool2(x2)

        #print('pool2', x1.shape, x2.shape)

        x = torch.cat((x1,x2), 1)

        x1 = self.branch1_conv3(x)
        x2 = self.branch2_conv3(x)

        #print('conv3', x1.shape, x2.shape)

        x1 = self.branch1_conv4(x1)
        x2 = self.branch2_conv4(x2)

        #print('conv4', x1.shape, x2.shape)

        x1 = self.branch1_conv5(x1)
        x2 = self.branch2_conv5(x2)

        #print('conv5', x1.shape, x2.shape)

        x1 = self.branch1_pool3(x1)
        x2 = self.branch2_pool3(x2)

        #print('pool3', x1.shape, x2.shape)

        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)

        #print('flatt', x1.shape, x2.shape)

        x = torch.cat((x1,x2), 1)

        x1 = self.branch1_fc1(x)
        x2 = self.branch2_fc1(x)

        #print('fc1  ', x1.shape, x2.shape)

        x = torch.cat((x1,x2), 1)

        x1 = self.branch1_fc2(x)
        x2 = self.branch2_fc2(x)

        #print('fc2  ', x1.shape, x2.shape)

        x = torch.cat((x1,x2), 1)

        x = self.fc(x)

        return x