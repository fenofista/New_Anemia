import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import random
# from dataset import KaggleDataset
import torchvision.transforms.functional as tx
from model import YOLOv1
from loss import Loss
import cv2
import os
import numpy as np
import math
from datetime import datetime
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt

import pandas as pd
from torch.utils.tensorboard import SummaryWriter
init_lr = 0.001
base_lr = 0.01
momentum = 0.9
weight_decay = 5.0e-4
num_epochs = 135
batch_size = 64

global_mean = [158.122433332762, 135.86361008407835, 118.4334553918209]
global_std = [58.57134455799642, 58.4258764351661, 59.034157439521465] 

few_data = False
data_num = 10
to_gray = False
def train_score(yolo):
#     yolo.load_state_dict(torch.load("../data/model/best_model.pth"))
    yolo.eval()
    pre_path = "../../../DL_data/competition2/datalab-2021-cup2-object-detection/"
    test_img_files = open("../data/train.txt")
    test_img_dir = "../data/image/"
    test_images = []
    mean_rgb = global_mean
    std_rgb = global_std
    mean = np.array(mean_rgb, dtype=np.float32)
    std = np.array(std_rgb, dtype=np.float32)
    image_size=448
    output_file = open('./train_prediction.txt', 'w')
    count = 0
    yolo.to("cuda")
    for line in test_img_files:
#         if count%500==0:
#             print(count)
        line = line.strip()
        ss = line.split(' ')
        img = cv2.imread(test_img_dir+ss[0])
        origin_shape = img.shape
    #     print(origin_shape)
        x_ratio = origin_shape[1]/image_size
        y_ratio = origin_shape[0]/image_size
    #     print(x_ratio, y_ratio)
        img = cv2.resize(img, dsize=(image_size, image_size), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img - mean) / std # normalize from -1.0 to 1.0.
        if to_gray:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = tx.to_tensor(img)
        img = img.to("cuda")
        if to_gray:
            img = img.reshape((1,1,image_size,image_size))
        else:
            img = img.reshape((1,3,image_size,image_size))
#         plt.imshow(img[0][0].cpu().detach().numpy())
#         plt.show()
#         print(img)
        prediction = yolo(img)
        prediction = prediction.to("cpu")
#         print(prediction)
        decode_pred = decode(prediction[0])
        
#         print(decode_pred)
    #     print(len(decode_pred[0]))
        for i in range(len(decode_pred[0])):
    #     for i in range(5):
    #         x_min = min(int(x_ratio*decode_pred[0][i][0]), int(x_ratio*decode_pred[0][i][2]))
    #         x_max = max(int(x_ratio*decode_pred[0][i][0]), int(x_ratio*decode_pred[0][i][2]))
    #         y_min = min(int(y_ratio*decode_pred[0][i][1]), int(y_ratio*decode_pred[0][i][3]))
    #         y_max = max(int(y_ratio*decode_pred[0][i][1]), int(y_ratio*decode_pred[0][i][3]))
#             print(decode_pred[0][i][0])
            decode_pred[0][i][0]=int(x_ratio*decode_pred[0][i][0])
            decode_pred[0][i][2]=int(x_ratio*decode_pred[0][i][2])
            decode_pred[0][i][1]=int(y_ratio*decode_pred[0][i][1])
            decode_pred[0][i][3]=int(y_ratio*decode_pred[0][i][3])
    #     print(decode_pred)
        s=ss[0]+" "
        for i in range(len(decode_pred[0])):
    #     for i in range(5):
            if decode_pred[2][i]<0.1:
                continue
            for j in range(4):
                s+=str(decode_pred[0][i][j].item())+" "
            s+=str(decode_pred[1][i].item())+" "
            s+=str(decode_pred[2][i].item())+" "
        s+="\n"
        output_file.write(s)
        count+=1
        if few_data:
            if count==data_num:
                break
    #     break
    output_file.close()
    import sys
#     sys.path.insert(0, pre_path+'./evaluate')
    import evaluate
    #evaluate.evaluate("input prediction file name", "desire output csv file name")
    evaluate.evaluate('./train_prediction.txt', './train_output_file.csv', "train", "../data/train.txt")
    train_csv = pd.read_csv("train_output_file.csv")
    MSE = 0
    for i in range(len(train_csv["packedCAP"])):
        MSE+=(1-train_csv["packedCAP"][i])**2
    return MSE



def valid_score(yolo):
#     yolo.load_state_dict(torch.load("../data/model/best_model.pth"))
    yolo.eval()
    pre_path = "../../../DL_data/competition2/datalab-2021-cup2-object-detection/"
    test_img_files = open("../data/valid.txt")
    test_img_dir = "../data/image/"
    test_images = []
    mean_rgb = global_mean
    std_rgb=global_std
    mean = np.array(mean_rgb, dtype=np.float32)
    std = np.array(std_rgb, dtype=np.float32)
    image_size=448
    output_file = open('./valid_prediction.txt', 'w')
    count = 0
    yolo.to("cuda")
    for line in test_img_files:
#         if count%500==0:
#             print(count)
        line = line.strip()
        ss = line.split(' ')
        img = cv2.imread(test_img_dir+ss[0])
        origin_shape = img.shape
    #     print(origin_shape)
        x_ratio = origin_shape[1]/image_size
        y_ratio = origin_shape[0]/image_size
    #     print(x_ratio, y_ratio)
        img = cv2.resize(img, dsize=(image_size, image_size), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img - mean) / std # normalize from -1.0 to 1.0.
        if to_gray:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = tx.to_tensor(img)
        img = img.to("cuda")
        if to_gray:
            img = img.reshape((1,1,image_size,image_size))
        else:
            img = img.reshape((1,3,image_size,image_size))

        prediction = yolo(img)
        prediction = prediction.to("cpu")

        decode_pred = decode(prediction[0])
    #     print(decode_pred)
    #     print(len(decode_pred[0]))
        for i in range(len(decode_pred[0])):
    #     for i in range(5):
    #         x_min = min(int(x_ratio*decode_pred[0][i][0]), int(x_ratio*decode_pred[0][i][2]))
    #         x_max = max(int(x_ratio*decode_pred[0][i][0]), int(x_ratio*decode_pred[0][i][2]))
    #         y_min = min(int(y_ratio*decode_pred[0][i][1]), int(y_ratio*decode_pred[0][i][3]))
    #         y_max = max(int(y_ratio*decode_pred[0][i][1]), int(y_ratio*decode_pred[0][i][3]))
            decode_pred[0][i][0]=int(x_ratio*decode_pred[0][i][0])
            decode_pred[0][i][2]=int(x_ratio*decode_pred[0][i][2])
            decode_pred[0][i][1]=int(y_ratio*decode_pred[0][i][1])
            decode_pred[0][i][3]=int(y_ratio*decode_pred[0][i][3])
    #     print(decode_pred)
        s=ss[0]+" "
        for i in range(len(decode_pred[0])):
    #     for i in range(5):
            if decode_pred[2][i]<0.1:
                continue
            for j in range(4):
                s+=str(decode_pred[0][i][j].item())+" "
            s+=str(decode_pred[1][i].item())+" "
            s+=str(decode_pred[2][i].item())+" "
        s+="\n"
        output_file.write(s)
        count+=1
        if few_data:
            if count==data_num:
                break
    #     break
    output_file.close()
    import sys
#     sys.path.insert(0, pre_path+'./evaluate')
    import evaluate
    #evaluate.evaluate("input prediction file name", "desire output csv file name")
    evaluate.evaluate('./valid_prediction.txt', './valid_output_file.csv', "valid", "../data/valid.txt")
    valid_csv = pd.read_csv("valid_output_file.csv")
    MSE = 0
    for i in range(len(valid_csv["packedCAP"])):
        MSE+=(1-valid_csv["packedCAP"][i])**2
    return MSE



def decode(pred_tensor):
        conf_thresh = 0
        prob_thresh = 0
        nms_thresh = 0.45
        S, B, C = 7, 2, 1
        boxes = []
        labels = []
        confidences = []
        class_scores = []

        cell_size = 1.0 / float(S)

        conf = pred_tensor[:, :, 4].unsqueeze(2) 
        for b in range(1, B):
            conf = torch.cat((conf, pred_tensor[:, :, 5*b + 4].unsqueeze(2)), 2)
        conf_mask = conf > conf_thresh 
        for i in range(S): 
            for j in range(S): 
                class_score, class_label = torch.max(pred_tensor[j, i, 5*B:], 0)
                for b in range(B):
                    conf = pred_tensor[j, i, 5*b + 4]
                    prob = conf * class_score
                    if float(prob) < prob_thresh:
                        continue
                    box = pred_tensor[j, i, 5*b : 5*b + 4]
                    corner_normalized = torch.FloatTensor([i, j]) * cell_size 
                    xy_normalized = box[:2] * cell_size + corner_normalized   
                    wh_normalized = box[2:]
                    box_xyxy = torch.FloatTensor(4) # [4,]
                    box_xyxy[:2] = xy_normalized - 0.5 * wh_normalized 
                    box_xyxy[2:] = xy_normalized + 0.5 * wh_normalized 
                    box_xyxy[:2] *= 448 
                    box_xyxy[2:] *= 448
                    
                    boxes.append(box_xyxy)
                    labels.append(class_label)
                    confidences.append(conf)
                    class_scores.append(class_score)

        if len(boxes) > 0:
            boxes = torch.stack(boxes, 0) 
            labels = torch.stack(labels, 0)             
            confidences = torch.stack(confidences, 0)   
            class_scores = torch.stack(class_scores, 0) 
        else:
            boxes = torch.FloatTensor(0, 4)
            labels = torch.LongTensor(0)
            confidences = torch.FloatTensor(0)
            class_scores = torch.FloatTensor(0)

        return boxes, labels, confidences, class_scores
    
    
    
# Learning rate scheduling.
def update_lr(optimizer, epoch, burnin_base, burnin_exp=4.0):
    if epoch == 0:
        lr = init_lr + (base_lr - init_lr) * math.pow(burnin_base, burnin_exp)
    elif epoch == 1:
        lr = base_lr
    elif epoch == 75:
        lr = 0.001
    elif epoch == 105:
        lr = 0.0001
    else:
        return

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']