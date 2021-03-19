# -*- coding: utf-8 -*-
"""
@Author : Horizon
@Date   : 2020-12-14 09:26:47
"""

import sys
from logger import Logger
from models.LoadModels import loadModels

#pytorch for deep neural network
import torch
import torchvision
from torchvision import models

import torchsummary
import torchscan
from thop import profile
from torchstat import stat

from torch.utils.tensorboard import SummaryWriter

#os for file processing
import os

#argparse for argument processing
import argparse

sys.stdout = Logger(filename="logs/model.log", stream=sys.stdout)

parser = argparse.ArgumentParser(description='check data')
parser.add_argument("--model",               dest="model",               default="ResNet",      type=str  )
parser.add_argument('--dataset_path',        dest='dataset_path',                               type=str  )
parser.add_argument('--batch_size',          dest='batch_size',          default=1,             type=int  )
parser.add_argument("--img_size",            dest="img_size",            default=224,           type=int  )
parser.add_argument("--show_in_tensorboard", dest="show_in_tensorboard", action="store_true")
args = parser.parse_args()

train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(args.dataset_path, 'train'))

# 类别名称
class_names = train_dataset.classes

model = loadModels(args.model, len(class_names), args.img_size)

input = torch.randn(args.batch_size, 3, args.img_size, args.img_size)

if(args.model != "densenet"):
    print('torchsummary result:')
    torchsummary.summary(model, (3, args.img_size, args.img_size), batch_size=args.batch_size, device="cpu")

if(args.model != 'GoogLeNet_aux'):
    print('torchscan result:')
    torchscan.summary(model, (3, args.img_size, args.img_size))

'''
print('PyTorch-OpCounter result:')
macs, params = profile(model, inputs=(input, ))
print('Total macc:{}, Total params: {}'.format(macs, params))

print('torchstat result:')
stat(model, (3, args.img_size, args.img_size))
'''
if args.show_in_tensorboard:
    writer = SummaryWriter()
    x = torch.rand(args.batch_size, 3, args.img_size, args.img_size)
    writer.add_graph(model, x)
    writer.close()

    print("Written model info to tensorboard. use command $tensorboard --logdir=runs$ to visualize")
