# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 15:16:23 2020

@author: Horizon
"""

#rich for pretty print
from rich import print
from rich.console import Console
from rich.table import Table
from rich.columns import Columns

#os for file processing
import os
import sys

#self-defined modules
import Horizon
from logger import Logger
from models.LoadModels import loadModels

#pytorch for deep neural network
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import torchvision
from torchvision.transforms import transforms

from torchsummary import summary

#numpy, pandas and matplotlib for data processing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#argparse for argument processing
import argparse

#time for time
import time

#tqdm for processing bar
from tqdm import tqdm

def get_args():
    
    parser = argparse.ArgumentParser(description='check data')
    parser.add_argument("--model",            dest="model",            default="ResNet",        type=str  )
    parser.add_argument("--pretrained",       dest="pretrained",       action="store_true"                )
    parser.add_argument("--frozen",           dest="frozen",           action="store_true"                )
    parser.add_argument('--dataset_path',     dest='dataset_path',                              type=str  )
    parser.add_argument('--batch_size',       dest='batch_size',       default=64,              type=int  )
    parser.add_argument("--epoch",            dest="epoch",            default=50,              type=int  )
    parser.add_argument("--optimizer",        dest="optimizer",        default="Adam",          type=str  )
    parser.add_argument("--lr",               dest="lr",               default=0.001,           type=float)
    parser.add_argument("--weight_decay",     dest="weight_decay",     default=0,               type=float)
    parser.add_argument("--img_size",         dest="img_size",         default=224,             type=int  )
    parser.add_argument("--checkpoint_path",  dest="checkpoint_path",                           type=str  )
    parser.add_argument('--save_interval',    dest='save_interval',    default=0,               type=int  )
    parser.add_argument('--save_path',        dest='save_path',        default='./checkpoints', type=str  )
    parser.add_argument("--crop",             dest="crop",             action="store_true")
    parser.add_argument("--showfig",          dest="showfig",          action="store_true")
    parser.add_argument("--confusion_matrix", dest="confusion_matrix", action="store_true")
    args = parser.parse_args()

    return args

def check_args(args):
    if(not args.dataset_path or not os.path.exists(args.dataset_path)):
        print("Dataset not exist!")
        quit()
    if(args.batch_size <= 0):
        print("Wrong batch size! Batch size should greater than 0")
        quit()
    if(args.epoch <= 0):
        print("Wrong epoch! Epoch should greater than 0")
        quit()
    if(args.lr <= 0):
        print("Wrong learning rate! Learning rate should greater than 0")
        quit()
    if(args.weight_decay < 0):
        print("Wrong weight decay! Weight decay should greater than 0")
        quit()
    if(args.img_size <= 0):
        print("Wrong image size! Image size should greater than 0")
        quit()
    if(args.checkpoint_path):
        if(not os.path.exists(args.checkpoint_path)):
            print("Checkpoint not exist!")
            quit()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

def print_params(args):

    console = Console()

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("model")
    table.add_column("dataset path")
    table.add_column("batch size")
    table.add_column("epoch")
    table.add_column("optimizer")
    table.add_column("learning rate")
    table.add_column("weight decay")
    table.add_column("image size")
    table.add_column("save interval")
    table.add_column("save path")
    
    if args.pretrained:
        if args.frozen:
            model_name = "pretrained_{}(frozen)".format(args.model)
        else:
            model_name = "pretrained_{}".format(args.model)
    else:
        model_name = args.model

    if args.save_interval == 0:
        save_interval_str = "Best"
    else:
        save_interval_str = str(args.save_interval)

    table.add_row(
        model_name,
        args.dataset_path,
        str(args.batch_size),
        str(args.epoch),
        args.optimizer,
        str(args.lr),
        str(args.weight_decay),
        str(args.img_size),
        save_interval_str,
        args.save_path,
    )

    console.print(table)

# training
def train(model, train_dataloader, epoch, loss_fc, caclu_confusion_matrix=False):

    train_loss    = 0.0
    train_correct = 0
    train_total   = len(train_dataloader.dataset)
    
    train_confusion_matrix = None

    if caclu_confusion_matrix:
        train_confusion_matrix = pd.DataFrame(data=np.zeros((len(class_names), len(class_names))),
                                            index = class_names, columns = class_names)
    
    bar = tqdm(total=len(train_dataloader), desc='Epoch {} Training  '.format(epoch + 1) ,ncols=180)

    for i, sample_batch in enumerate(train_dataloader):
        inputs = sample_batch[0]
        labels = sample_batch[1]

        model.train()

        # GPU/CPU
        if torch.cuda.is_available():
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        optimizer.zero_grad()

        # forward
        outputs = model(inputs)

        # loss
        loss = loss_fc(outputs, labels)
        '''
        loss1 = loss_fc(outputs, labels)
        loss2 = loss_fc(aux1, labels)
        loss3 = loss_fc(aux2, labels)
        loss = loss1 + 0.3*(loss2+loss3)
        '''

        # backward
        loss.backward()

        # update
        optimizer.step()

        _, prediction = torch.max(outputs, 1)
        correct = (torch.sum((prediction == labels))).item()
        train_correct += correct

        # total loss
        current_loss = loss.item()
        train_loss += current_loss
        
        if caclu_confusion_matrix:
            for class_index in range(len(class_names)):
                
                index_labels = prediction[labels == class_index]
                
                for index_label in index_labels:
                    
                    train_confusion_matrix[class_names[class_index]][int(index_label)] += 1
        
        bar.set_postfix(acc='{:.5f}'.format(correct / int(labels.shape[0])), loss='{:5f}'.format(current_loss))
        bar.update(1)

    bar.close()

    return train_loss, train_total, train_correct, train_confusion_matrix

# validating
def evaluate(model, val_dataloader, epoch, loss_fc, caclu_confusion_matrix=False):

    val_loss      = 0.0
    val_correct   = 0
    val_total     = len(val_dataloader.dataset)

    val_confusion_matrix = None
    
    if caclu_confusion_matrix:
        val_confusion_matrix = pd.DataFrame(data=np.zeros((len(class_names), len(class_names))),
                                            index = class_names, columns = class_names)
    
    bar = tqdm(total=len(val_dataloader), desc='Epoch {} Validating'.format(epoch + 1), ncols=180)

    model.eval()
    with torch.no_grad():
        for images_test, labels_test in val_dataloader:
            if torch.cuda.is_available():
                images_test = images_test.cuda(non_blocking=True)
                labels_test = labels_test.cuda(non_blocking=True)

            outputs_test = model(images_test)

            loss = loss_fc(outputs_test, labels_test)
            current_loss = loss.item()
            val_loss += current_loss

            _, prediction = torch.max(outputs_test, 1)
            correct = (torch.sum((prediction == labels_test))).item()
            val_correct += correct
            
            if caclu_confusion_matrix:
                for class_index in range(len(class_names)):
                    
                    index_labels = prediction[labels_test == class_index]
                    
                    for index_label in index_labels:
                        
                        val_confusion_matrix[class_names[class_index]][int(index_label)] += 1
            
            bar.set_postfix(acc='{:.5f}'.format(correct / int(labels_test.shape[0])), loss='{:5f}'.format(current_loss))
            bar.update(1)

    bar.close()

    return val_loss, val_total, val_correct, val_confusion_matrix

if(__name__ == "__main__"):

    torch.backends.cudnn.benchmark = True

    Horizon.print_horizon("lean")

    sys.stdout = Logger(filename="logs/train.log", stream=sys.stdout)

    print("Checking      ......")

    args = get_args()

    check_args(args)

    print("Initializaion ......")

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        )

    #prepare data
    train_traisform = transforms.Compose([
                                        transforms.RandomResizedCrop(args.img_size),
                                        #transforms.Resize((args.img_size,args.img_size)),
                                        transforms.RandomHorizontalFlip(),  
                                        #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),                                      
                                        transforms.ToTensor(),
                                        normalize,
                                    ])

    if args.crop:
        val_traisform = transforms.Compose([
                                            transforms.Resize(args.img_size),
                                            transforms.CenterCrop(args.img_size),
                                            #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                            transforms.ToTensor(),                                            
                                            normalize,
                                        ])
    else:
        val_traisform = transforms.Compose([
                                            transforms.Resize((args.img_size, args.img_size)),
                                            #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                            transforms.ToTensor(),
                                            normalize,
                                        ])

    train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(args.dataset_path, 'train'), transform=train_traisform)
    val_dataset   = torchvision.datasets.ImageFolder(root=os.path.join(args.dataset_path, 'val'),   transform=val_traisform)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True, drop_last=True)
    val_dataloader   = DataLoader(dataset=val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # classes
    class_names = train_dataset.classes

    print("Loading model ......")

    # -------------------------Model, loss----------------------
    model = loadModels(args.model, len(class_names), args.img_size, args.pretrained, args.frozen)

    if args.model != "densenet":
        summary(model, (3, args.img_size, args.img_size), batch_size=args.batch_size, device="cpu")

    # 模型迁移到CPU/GPU
    if torch.cuda.is_available():
        model = model.cuda()

    # 定义损失函数
    loss_fc = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        loss_fc = loss_fc.cuda()

    # 选择优化器
    if args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    else:
        print("Unknown optimizer!")
        quit()

    #scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    print_params(args)

    print("Class names:")
    print(Columns(class_names))

    print("Starting training")

    # ----------------Training-----------------
    if args.checkpoint_path:
        print("Loading checkpoint from {}".format(args.checkpoint_path))
        model.load_state_dict(torch.load(args.checkpoint_path))
        val_loss, val_total, val_correct, _ = evaluate(model, val_dataloader, epoch=0, loss_fc=loss_fc, caclu_confusion_matrix=args.confusion_matrix)
        best_acc = val_correct / val_total
        best_loss = val_loss / len(val_dataloader)
        print("Acc from checkpoint is {:5f}, loss is {:5f}".format(best_acc, best_loss))

    else:
        best_acc = 0
        best_loss = 0
    
    best_acc_epoch = 0
    best_loss_epoch = 0
    max_loss = 0

    train_losses = []
    validation_losses = []
    train_accs = []
    validation_accs = []

    for epoch in range(args.epoch):

        start_time = time.time()

        train_loss, train_total, train_correct, train_confusion_matrix = train(model, train_dataloader, epoch, loss_fc, args.confusion_matrix)
        val_loss, val_total, val_correct, val_confusion_matrix = evaluate(model, val_dataloader, epoch, loss_fc, args.confusion_matrix)
        
        #scheduler.step()

        end_time = time.time()
        
        print()
        print('Epoch {}: Time cost: {:.2f}'.format(epoch + 1, end_time - start_time))
        print('Train loss:      {:5f}, Train acc:      {:.5f} ({}/{})'.format(train_loss/len(train_dataloader), train_correct/train_total, 
                                                                train_correct, train_total))
        
        print('Validation loss: {:5f}, Validation acc: {:.5f} ({}/{})'.format(val_loss/len(val_dataloader), val_correct/val_total,
                                                                        val_correct, val_total))
        
        train_losses.append(train_loss/len(train_dataloader))
        validation_losses.append(val_loss/len(val_dataloader))
        train_accs.append(train_correct/train_total)
        validation_accs.append(val_correct/val_total)

        if args.confusion_matrix:
            print('Confusion matrix:')
            print('Training set:')
            print(train_confusion_matrix)
            print('Validation set:')
            print(val_confusion_matrix)

        if(best_acc < val_correct / val_total):
            best_acc = val_correct / val_total
            best_acc_epoch = epoch + 1
            start_time = time.time()
            torch.save(model.state_dict(), os.path.join(args.save_path, args.model + '_model_{}.pth'.format('best_acc')))
            end_time = time.time()
            print('Saved best acc model!  Save cost: {:5f}. Best acc :{:5f}. '.format(end_time - start_time, best_acc))
        
        if(best_loss == 0):
            best_loss = val_loss / len(val_dataloader)
            best_loss_epoch = epoch + 1
            start_time = time.time()
            torch.save(model.state_dict(), os.path.join(args.save_path, args.model + '_model_{}.pth'.format('best_loss')))
            end_time = time.time()
            print('Saved best loss model! Save cost: {:5f}. Best loss:{:5f}. '.format(end_time - start_time, best_loss))
        elif(best_loss > val_loss / len(val_dataloader)):
            best_loss = val_loss / len(val_dataloader)
            best_loss_epoch = epoch + 1
            start_time = time.time()
            torch.save(model.state_dict(), os.path.join(args.save_path, args.model + '_model_{}.pth'.format('best_loss')))
            end_time = time.time()
            print('Saved best loss model! Save cost: {:5f}. Best loss:{:5f}. '.format(end_time - start_time, best_loss))
        
        if(args.save_interval != 0 and (epoch + 1) % args.save_interval == 0):
            start_time = time.time()
            torch.save(model.state_dict(), os.path.join(args.save_path, args.model + '_model_{}.pth'.format(epoch + 1)))
            end_time = time.time()
            print('Saved epoch {} model!. Save cost:{:5f}'.format(epoch + 1, end_time - start_time))
        
        if(train_loss/len(train_dataloader) > max_loss):
            max_loss = train_loss/len(train_dataloader)
        if(val_loss/len(val_dataloader)  > max_loss):
            max_loss = val_loss/len(val_dataloader)

        plt.figure(1, figsize=(15,7))
        plt.cla()
        plt.clf()
        x = range(1, len(train_losses)+1)

        plt.subplot(1,2,1)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.xlim((0,args.epoch)) 
        plt.ylim([0, float('%.1g' % (2*max_loss))])
        plt.scatter(x, train_losses,label="train", s=9)
        plt.scatter(x, validation_losses,label="validation", s=9)
        plt.legend()    

        plt.subplot(1,2,2)
        plt.xlabel("epoch")
        plt.ylabel("acc")
        plt.xlim((0,args.epoch)) 
        plt.ylim((0.0,1.0))    
        plt.scatter(x, train_accs,label="train", s=9)
        plt.scatter(x, validation_accs,label="validation", s=9) 
        plt.legend()   
        
        plt.savefig('result.jpg')
        plt.close()

        train_loss = 0.0
        val_loss = 0.0
        
        print()

    print('Training finish! Best acc is {:5f}, which appears in epoch {}; Best loss is {:5f}, which appears in epoch {};'.format(best_acc, best_acc_epoch, best_loss, best_loss_epoch))

    if args.showfig:
        plt.show()

