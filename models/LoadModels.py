from .AlexNet import loadAlexNet, AlexNet_models
from .VGG import loadVGG, VGGmodels, VGG_path
from .GoogLeNet import loadGoogLeNet, GoogLeNetmodels
from .ResNet import loadResNet, ResNet_models
from .DenseNet import loadDenseNet, DenseNet_models
from .SimpleNet import SimpleNet96

import torch
import torch.nn as nn
from torchvision import models

def loadModels(model_name="ResNet18", num_classes=1000, img_size=224, pretrained=False, frozen=False):

    #My models
    if(model_name in VGGmodels):
        
        if pretrained:
            #先使用1000类分类器来加载预训练模型
            model = loadVGG(model_name, num_classes=1000, img_size=img_size)
            model_dict = model.state_dict()
            pretrained_dict = torch.load(VGG_path[model_name])
            pretrained_values = pretrained_dict.values()
            temp_dict = model_dict.copy()
            #将pytorch预训练模型参数的键改为我自己模型中的键
            if model_name == "VGG16_conv1":
                [temp_dict.pop(k) for k in ['conv3.4.weight', 'conv3.4.bias', 'conv4.4.weight', 'conv4.4.bias', 'conv5.4.weight', 'conv5.4.bias']]
            
            pretrained_dict = dict(zip(temp_dict.keys(), pretrained_values))
            model_dict.update(pretrained_dict)

            #加载参数
            model.load_state_dict(model_dict)

            # 全连接层的输入通道in_channels个数
            num_fc_in = model.fc[-1].in_features

            # 改变全连接层
            model.fc[-1] = nn.Linear(num_fc_in, num_classes)

            if frozen:
                for k, v in model.named_parameters():
                    if('conv' in k and k in temp_dict.keys()):
                        v.requires_grad = False
        
        else:
            model = loadVGG(model_name, num_classes=num_classes, img_size=img_size)

    elif(model_name in AlexNet_models):
        model = loadAlexNet(model_name, num_classes=num_classes, img_size=img_size)

    elif(model_name in GoogLeNetmodels):
        model = loadGoogLeNet(model_name, num_classes=num_classes, img_size=img_size)
    
    elif(model_name in ResNet_models):
        model = loadResNet(model_name, num_classes=num_classes, img_size=img_size)

    elif(model_name in DenseNet_models):
        model = loadDenseNet(model_name, num_classes=num_classes, img_size=img_size)

    elif(model_name == "SimpleNet"):
        model = SimpleNet96(num_classes=num_classes)

    #Pytorch models
    elif(model_name == "alexnet"):
        model = models.alexnet(pretrained=pretrained)

        if frozen:
            for k, v in model.named_parameters():
                v.requires_grad = False        

        num_fc_in = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_fc_in, num_classes)

    elif(model_name == "googlenet"):
        model = models.googlenet(pretrained=pretrained, aux_logits=True)

        if frozen:
            for k, v in model.named_parameters():
                v.requires_grad = False        

        num_fc_in = model.fc.in_features
        model.fc = nn.Linear(num_fc_in, num_classes)

    elif(model_name == "vgg"):
        model = models.vgg16(pretrained=pretrained)

        if frozen:
            for k, v in model.named_parameters():
                v.requires_grad = False        
        
        num_fc_in = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_fc_in, num_classes)

        #model.avgpool = nn.AdaptiveAvgPool2d((1,1))
        #model.classifier = nn.Linear(512, num_classes)


    elif(model_name == "resnet"):
        model = models.resnet18(pretrained=pretrained)

        if frozen:
            for k, v in model.named_parameters():
                v.requires_grad = False        
        
        num_fc_in = model.fc.in_features
        model.fc = nn.Linear(num_fc_in, num_classes)

    elif(model_name == "densenet"):
        model = models.densenet161(pretrained=pretrained)

        if frozen:
            for k, v in model.named_parameters():
                v.requires_grad = False        
        
        num_fc_in = model.classifier.in_features
        model.classifier = nn.Linear(num_fc_in, num_classes)
    
    else:
        print("Undefined model!")
        quit()
    
    return model