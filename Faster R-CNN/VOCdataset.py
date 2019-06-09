#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 21:25:58 2019

@author: ziyan
"""
import os
import pandas as pd
import torch
import torch.utils.data as td
import torchvision as tv
from PIL import Image
import xml.etree.ElementTree as ET

class VOCDataset(td.Dataset):
    def __init__(self, root_dir, mode='train', image_size=(800, 800)): 
        super(VOCDataset, self).__init__()
        self.mode = mode
        self.image_size = image_size
        
        self.images_dir = os.path.join(root_dir, 'JPEGImages')
        self.images_idx = pd.read_csv(os.path.join(root_dir, 'ImageSets/Main' ,"%s.txt" % mode), sep=' ', header=None, error_bad_lines=False) 
        self.ann_dir    = os.path.join(root_dir, 'Annotations')
        
        self.labels_map = {'aeroplane':20,  'bicycle':1, 'bird':2,  'boat':3,      'bottle':4, 
                       'bus':5,        'car':6,      'cat':7,  'chair':8,     'cow':9,
                       'diningtable':10,'dog':11,    'horse':12,  'motorbike':13, 'person':14,
                       'pottedplant':15,'sheep':16,  'sofa':17,   'train':18,   'tvmonitor':19, 'background':0}

    def __len__(self):
        # Return the size of the dataset
        return len(self.images_idx)
    def __repr__(self):
        # Return the data is training set or testing set, and its image size
        return "VOCDataset(mode={}, image_size={})". \
            format(self.mode, self.image_size)
    
    def __getitem__(self, idx, name=[]):
        # Return the preprocessed data (tensor) and labels.
        if len(name)>0:
            img_path = os.path.join(self.images_dir, name+".jpg") 
            target = self.get_obj(name)
        else:
            img_path = os.path.join(self.images_dir, self.images_idx.iloc[idx][0]+".jpg") 
            target = self.get_obj(self.images_idx.iloc[idx][0])
        img = Image.open(img_path).convert('RGB')
        
        target = self.encode(target)
        
        transform = tv.transforms.Compose([ 
            # resize the image
            tv.transforms.Resize(self.image_size), 
            # convert a PIL Image to tensor in range [0,1]
            tv.transforms.ToTensor(), 
            # normalize the tensor to [-1,1]
            #tv.transforms.Normalize((1/2,1/2,1/2),(1/2,1/2,1/2)) 
        ])
        # Transform
        img = transform(img)

        return img, target
    
    def get_obj(self, img_name):
        
        img = {'object':[]} 
        tree = ET.parse(os.path.join(self.ann_dir, img_name+".xml"))
     
        for elem in tree.iter():
            #print(elem.tag)
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
                
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
                
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}
                for attr in list(elem):
                    
                    if 'name' in attr.tag:  
                        obj['name'] = attr.text
                        
                        img['object'] += [obj]  
                           
                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:       
                                obj['xmin'] = float(dim.text)
                            if 'ymin' in dim.tag:      
                                obj['ymin'] = float(dim.text)
                            if 'xmax' in dim.tag:           
                                obj['xmax'] = float(dim.text)
                            if 'ymax' in dim.tag:
                                obj['ymax'] = float(dim.text)
                                                                              
        return img
    
    def encode(self, objs):
        res = {'boxes':[],'labels':[]}
        frac_x = self.image_size[0] / objs['width']
        frac_y = self.image_size[0] / objs['height']
        
        for i in range(len(objs['object'])):
            if objs['object'][i]['name'] in self.labels_map.keys():
                res['labels'].append(self.labels_map[objs['object'][i]['name']])
                res['boxes'].append([objs['object'][i]['xmin']*frac_x, 
                                    objs['object'][i]['ymin']*frac_y,
                                    objs['object'][i]['xmax']*frac_x,
                                    objs['object'][i]['ymax']*frac_y])

        
        res['labels'] = torch.Tensor(res['labels']).int()
        res['boxes'] = torch.Tensor(res['boxes'])
        return res
    