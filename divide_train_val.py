# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 14:48:17 2020

@author: Horizon
"""

import os
import argparse
from tqdm import tqdm
import random

if(__name__ == '__main__'):
    parser = argparse.ArgumentParser(description='check data')
    parser.add_argument('--initial_dataset', dest='initial_dataset', default='..\ObjectCategories', type=str)
    parser.add_argument('--train_percentage', dest='train_percentage', default=80, type=int)
    parser.add_argument('--output', dest='output', default='..\Caltech-256', type=str)
    args = parser.parse_args()
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    if not os.path.exists(os.path.join(args.output, 'train')):
        os.makedirs(os.path.join(args.output, 'train'))
    if not os.path.exists(os.path.join(args.output, 'val')):
        os.makedirs(os.path.join(args.output, 'val'))
    
    folder_list = os.listdir(args.initial_dataset)
    
    for folder in folder_list:
        if not os.path.exists(os.path.join(os.path.join(args.output, 'train'), folder)):
           os.makedirs(os.path.join(os.path.join(args.output, 'train'), folder))
        if not os.path.exists(os.path.join(os.path.join(args.output, 'val'), folder)):
           os.makedirs(os.path.join(os.path.join(args.output, 'val'), folder))
    
    for folder in folder_list:
        
        folder_path = os.path.join(args.initial_dataset, folder)
        
        images = os.listdir(folder_path)
        
        image_num = len(images)
        
        val_percentage = 1.0 - args.train_percentage / 100.0
        
        bar = tqdm(total=len(images))
        bar.set_description(folder)
        
        for image in images:
            
            image_path = os.path.join(os.path.join(args.initial_dataset, folder), image)
            
            probability = random.random()
            
            if(probability < val_percentage):
                out_path = os.path.join(os.path.join(args.output, 'val'), os.path.join(folder, image))
                os.popen('copy {} {}'.format(image_path, out_path))                
            
            else:
                out_path = os.path.join(os.path.join(args.output, 'train'), os.path.join(folder, image))
                os.popen('copy {} {}'.format(image_path, out_path))   
            
            bar.update(1)
        
        bar.close()
            
            
