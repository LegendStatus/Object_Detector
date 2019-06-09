# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 16:36:32 2019

@author: Lance
"""

from collections import defaultdict
from tqdm import tqdm
import cv2
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

def AP(recall, precision):
    '''Compute average precision for one class'''
    rec = np.concatenate(([0.], recall, [1.]))
    prec = np.concatenate(([0.], precision, [0.]))
    for i in range(prec.size -1, 0, -1):
        prec[i-1] = np.maximum(prec[i-1],prec[i])
    i = np.where(rec[1:] != rec[:-1])[0]
    ap = np.sum((rec[i + 1] - rec[i]) * prec[i + 1])
    return ap

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

def evaluate(preds,target,VOC_CLASSES=VOC_CLASSES,threshold=0.5):
    '''
    preds = {'cat':[[image_id,confidence,x1,y1,x2,y2],...],'dog':[[],...]}
    target = {(image_id,class):[[],]}
    threshold = IoU threshold for true positives 
    '''
    aps = []
    recall = []
    precision = []
    for i,class_ in enumerate(VOC_CLASSES):
        pred = preds[class_]   # [[image_id,confidence,x1,y1,x2,y2],...]
        
        # When there is no detection in the image
        if len(pred) == 0: 
            ap = 0
            print('---class {} ap {}---'.format(class_,ap))
            aps += [ap]
            continue
            
        image_ids = [x[0] for x in pred]
        confidence = np.array([float(x[1]) for x in pred])
        BB = np.array([x[2:] for x in pred])  # bounding box
        # Sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # Go through all images in the class and mark TPs and FPs
        npos = 0.
        for (key1,key2) in target:
            if key2 == class_:
                npos += len(target[(key1,key2)])  # make sure no examples are lost
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d,image_id in enumerate(image_ids):
            bb = BB[d]  # predicting bounding box
            if (image_id,class_) in target:
                BBGT = target[(image_id,class_)]  # ground truth bounding box
                for bbgt in BBGT:
                    # compute overlaps and intersections
                    ixmin = np.maximum(bbgt[0], bb[0])
                    iymin = np.maximum(bbgt[1], bb[1])
                    ixmax = np.minimum(bbgt[2], bb[2])
                    iymax = np.minimum(bbgt[3], bb[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih
                    union = (bb[2]-bb[0]+1.)*(bb[3]-bb[1]+1.) + (bbgt[2]-bbgt[0]+1.)*(bbgt[3]-bbgt[1]+1.) - inters
                    if union == 0:  
                        continue
                    # based on IoU to mark TP and FP
                    overlaps = inters/union
                    if overlaps > threshold:
                        tp[d] = 1
                        BBGT.remove(bbgt)  # Bounding box has ben detected
                        if len(BBGT) == 0:
                            del target[(image_id,class_)]  # Delete key value of examples without bounding box
                        break
                fp[d] = 1-tp[d]
            else:
                fp[d] = 1
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp/float(npos) 
        recall.append(rec)  
        prec = tp/np.maximum(tp + fp, np.finfo(np.float64).eps)
        precision.append(prec)
        # Compute average precision
        ap = AP(rec, prec)
        print('---class {} ap {}---'.format(class_,ap))
        aps += [ap]
    print('---map {}---'.format(np.mean(aps)))
    return recall, precision, aps


def cal_devidedclass(val_set,model,device):
    model.eval()
    class_gt_dict = {}
    class_pr_dict = {}
    labels_map = {0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 4: 'bottle', 5: 'bus', 6: 'car', 7: 'cat',
                  8: 'chair', 9: 'cow', 10: 'diningtable', 11: 'dog', 12: 'horse', 13: 'motorbike', 14: 'person',
                  15: 'pottedplant', 16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tvmonitor', 20: 'background'}
    for labe in labels_map.values():
        class_pr_dict[labe] = []
    for i in range(len(val_set)):
        if i %200 == 0: print(i)
        images, targets = val_set[i]
        images = images.to(device)
        
        targets['boxes'] = targets['boxes'].to(device).to('cpu').tolist()
        targets['labels'] = targets['labels'].to(device).to('cpu').tolist()
        temlab = targets['labels']
        ntemlab = [labels_map[ele] for ele in temlab]
        targets['labels'] = ntemlab
        ntar = starget(targets)
        for key,box in ntar.items():
            class_gt_dict[(i,key)] = box
        
        
        prediction = model([images])
        
        
        tem = prediction[0]['labels'].to('cpu').tolist()
        tem = [labels_map[ele] for ele in tem]
        prediction[0]['labels'] = tem
        npre = spre(prediction[0],i)
        for key,id_scores_box in npre.items():
            if key not in class_pr_dict.keys():
                class_pr_dict[key] = []
            class_pr_dict[key].extend(id_scores_box)
    return class_gt_dict,class_pr_dict

def starget(tar):
    andic = {}
    boxes = tar['boxes']
    lab = tar['labels']
    for i in range(len(lab)):
        if lab[i] not in andic.keys():
            andic[lab[i]] = [boxes[i]]
        else:
            andic[lab[i]].append(boxes[i])
    return andic
        
def spre(pre,imid):
    labels = pre['labels']
    boxes = pre['boxes'].to('cpu').tolist()
    scores = pre['scores'].to('cpu').tolist()
    cur_dict = {cur_label : []  for cur_label in set(labels)}
    for key in cur_dict.keys():
        cur_dict[key] = []
    for i in range(len(boxes)):
        temlist = [imid,scores[i],boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3]]
        cur_dict[labels[i]].append(temlist)
    return cur_dict
def run(model,val_set,device,VOC_CLASSES=VOC_CLASSES,thr = 0.5):
    tic = time.time()
    class_gt_dict,class_pr_dict = cal_devidedclass(val_set,model,device)
    recall, precision, aps = evaluate(class_pr_dict,class_gt_dict,VOC_CLASSES=VOC_CLASSES,threshold=thr)
    toc = time.time()
    print('time cost', toc-tic,'s')
    return recall, precision, aps