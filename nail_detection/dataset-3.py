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
global_mean = [158.122433332762, 135.86361008407835, 118.4334553918209]
global_std = [58.57134455799642, 58.4258764351661, 59.034157439521465] 
few_data = False
data_num = 10
to_gray = False
class MyDataset():
    def __init__(self, augment, image_dir, label_txt, image_size=448, grid_size=7, num_bboxes=2, num_classes=1):
        self.image_size = image_size
        self.augment = augment
        self.S = grid_size
        self.B = num_bboxes
        self.C = num_classes
        self.cach={}
        mean_rgb = global_mean
        std_rgb = global_std
        self.mean = np.array(mean_rgb, dtype=np.float32)
        self.std = np.array(std_rgb, dtype=np.float32)
        count = 0
#         self.to_tensor = transforms.ToTensor()
        self.paths, self.boxes, self.labels = [], [], []
        with open(label_txt) as f:
            lines = f.readlines()
        
        for line in lines:
            splitted = line.strip().split()

            fname = splitted[0]
            path = os.path.join(image_dir, fname)
            self.paths.append(path)

            num_boxes = (len(splitted) - 1) // 5
            box, label = [], []
            for i in range(num_boxes):
                x1 = float(splitted[5*i + 1])
                y1 = float(splitted[5*i + 2])
                x2 = float(splitted[5*i + 3])
                y2 = float(splitted[5*i + 4])
                c  =   int(splitted[5*i + 5])
                box.append([x1, y1, x2, y2])
                label.append(c)
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))
            count+=1
            if few_data:
                if count==data_num:
                    break

        self.num_samples = len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = self.cach[path]
        except:
            img = cv2.imread(path)
            self.cach[path] = img
        boxes = self.boxes[idx].clone() # [n, 4]
        labels = self.labels[idx].clone() # [n,]

        if self.augment:
            img, boxes = self.random_flip_lr(img, boxes)
            img, boxes = self.random_flip_ud(img, boxes)
            img, boxes = self.random_scale(img, boxes)
            img = self.random_blur(img)
#             img = self.random_brightness(img)
#             imgw = self.random_hue(img)
#             img = self.random_saturation(img)
#             img = self.random_noise(img)
            img, boxes, labels = self.random_shift(img, boxes, labels)
            img, boxes, labels = self.random_crop(img, boxes, labels)

        h, w, _ = img.shape
        boxes /= torch.Tensor([[w, h, w, h]]).expand_as(boxes) # normalize (x1, y1, x2, y2) w.r.t. image width/height.
        target = self.encode(boxes, labels) # [S, S, 5 x B + C]

        img = cv2.resize(img, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img - self.mean) / self.std # normalize from -1.0 to 1.0.
        if to_gray:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = tx.to_tensor(img)
        
        return (img, target)

    def __len__(self):
        return self.num_samples
    
    def random_noise(self, img):
        if random.random() < 0.5:
            return img

        mean = 0
        sigma = 0.1

        noise = np.random.normal(mean, sigma, img.shape)
        noise = (noise*255)
        noise = noise.astype(np.uint8)
        img = img + noise

        return img
    def random_flip_lr(self, img, boxes):
        if random.random() < 0.2:
            return img, boxes
#         print("flip")
        h, w, _ = img.shape

        img = np.fliplr(img)

        x1 = boxes[:, 0]
        x2 =  boxes[:, 2]
        x1_new = w - x2
        x2_new = w - x1
        boxes[:, 0] = x1_new
        boxes[:, 2] =  x2_new

        return img, boxes
    def random_flip_ud(self, img, boxes):
        
        if random.random() < 0.2:
            return img, boxes
#         print("flip")
        h, w, _ = img.shape

        img = np.flipud(img)

        y1 = boxes[:, 1]
        y2 =  boxes[:, 3]
        y1_new = h - y2
        y2_new = h - y1
        boxes[:, 1] = y1_new
        boxes[:, 3] = y2_new

        return img, boxes


    def random_scale(self, img, boxes):
        if random.random() < 0.2:
            return img, boxes
#         print("scale")
        scale = random.uniform(0.8, 1.2)
        h, w, _ = img.shape
        img = cv2.resize(img, dsize=(int(w * scale), h), interpolation=cv2.INTER_LINEAR)

        scale_ratio = torch.FloatTensor([[scale, 1.0, scale, 1.0]]).expand_as(boxes)
        boxes = boxes * scale_ratio

        return img, boxes

    def random_blur(self, bgr):
        if random.random() < 0.5:
            return bgr
#         print("blur")
        ksize = random.choice([2, 3, 4, 5, 6, 7])
        bgr = cv2.blur(bgr, (ksize, ksize))
        return bgr

    def random_brightness(self, bgr):
        if random.random() < 0.5:
            return bgr
#         print("brightness")
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        adjust = random.uniform(0.5, 1.5)
        v = v * adjust
        v = np.clip(v, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h, s, v))
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return bgr

    def random_hue(self, bgr):
        if random.random() < 0.5:
            return bgr

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        adjust = random.uniform(0.5, 1.5)
        h = h * adjust
        h = np.clip(h, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h, s, v))
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return bgr

    def random_saturation(self, bgr):
        if random.random() < 0.5:
            return bgr
#         print("saturation")
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        adjust = random.uniform(0.5, 1.5)
        s = s * adjust
        s = np.clip(s, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h, s, v))
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return bgr
    def random_shift(self, img, boxes, labels):
        if random.random() < 0.2:
            return img, boxes, labels

        center = (boxes[:, 2:] + boxes[:, :2]) / 2.0

        h, w, c = img.shape
        img_out = np.zeros((h, w, c), dtype=img.dtype)
        mean_bgr = self.mean[::-1]
        img_out[:, :] = mean_bgr

        dx = random.uniform(-w*0.2, w*0.2)
        dy = random.uniform(-h*0.2, h*0.2)
        dx, dy = int(dx), int(dy)

        if dx >= 0 and dy >= 0:
            img_out[dy:, dx:] = img[:h-dy, :w-dx]
        elif dx >= 0 and dy < 0:
            img_out[:h+dy, dx:] = img[-dy:, :w-dx]
        elif dx < 0 and dy >= 0:
            img_out[dy:, :w+dx] = img[:h-dy, -dx:]
        elif dx < 0 and dy < 0:
            img_out[:h+dy, :w+dx] = img[-dy:, -dx:]

        center = center + torch.FloatTensor([[dx, dy]]).expand_as(center) 
        mask_x = (center[:, 0] >= 0) & (center[:, 0] < w) 
        mask_y = (center[:, 1] >= 0) & (center[:, 1] < h) 
        mask = (mask_x & mask_y).view(-1, 1) 

        boxes_out = boxes[mask.expand_as(boxes)].view(-1, 4) 
        if len(boxes_out) == 0:
            return img, boxes, labels
        shift = torch.FloatTensor([[dx, dy, dx, dy]]).expand_as(boxes_out) 

        boxes_out = boxes_out + shift
        boxes_out[:, 0] = boxes_out[:, 0].clamp_(min=0, max=w)
        boxes_out[:, 2] = boxes_out[:, 2].clamp_(min=0, max=w)
        boxes_out[:, 1] = boxes_out[:, 1].clamp_(min=0, max=h)
        boxes_out[:, 3] = boxes_out[:, 3].clamp_(min=0, max=h)

        labels_out = labels[mask.view(-1)]

        return img_out, boxes_out, labels_out

    def random_crop(self, img, boxes, labels):
        if random.random() < 0.2:
            return img, boxes, labels
#         print("crop")
        center = (boxes[:, 2:] + boxes[:, :2]) / 2.0

        h_orig, w_orig, _ = img.shape
        h = random.uniform(0.6 * h_orig, h_orig)
        w = random.uniform(0.6 * w_orig, w_orig)
        y = random.uniform(0, h_orig - h)
        x = random.uniform(0, w_orig - w)
        h, w, x, y = int(h), int(w), int(x), int(y)

        center = center - torch.FloatTensor([[x, y]]).expand_as(center) 
        mask_x = (center[:, 0] >= 0) & (center[:, 0] < w) 
        mask_y = (center[:, 1] >= 0) & (center[:, 1] < h) 
        mask = (mask_x & mask_y).view(-1, 1) 

        boxes_out = boxes[mask.expand_as(boxes)].view(-1, 4) 
        if len(boxes_out) == 0:
            return img, boxes, labels
        shift = torch.FloatTensor([[x, y, x, y]]).expand_as(boxes_out) 

        boxes_out = boxes_out - shift
        boxes_out[:, 0] = boxes_out[:, 0].clamp_(min=0, max=w)
        boxes_out[:, 2] = boxes_out[:, 2].clamp_(min=0, max=w)
        boxes_out[:, 1] = boxes_out[:, 1].clamp_(min=0, max=h)
        boxes_out[:, 3] = boxes_out[:, 3].clamp_(min=0, max=h)

        labels_out = labels[mask.view(-1)]
        img_out = img[y:y+h, x:x+w, :]

        return img_out, boxes_out, labels_out
    
    def encode(self, boxes, labels):
        N = 5 * self.B + self.C

        target = torch.zeros(self.S, self.S, N)
        cell_size = 1.0 / float(self.S)
        boxes_wh = boxes[:, 2:] - boxes[:, :2] 
        boxes_xy = (boxes[:, 2:] + boxes[:, :2]) / 2.0 
        for b in range(boxes.size(0)):
            xy, wh, label = boxes_xy[b], boxes_wh[b], int(labels[b])

            ij = (xy / cell_size).ceil() - 1.0
            i, j = int(ij[0]), int(ij[1])
            corner = ij * cell_size 
            xy_normalized = (xy - corner) / cell_size 

            for k in range(self.B):
                s = 5 * k
                target[j, i, s  :s+2] = xy_normalized
                target[j, i, s+2:s+4] = wh
                target[j, i, s+4    ] = 1.0
            target[j, i, 5*self.B + label] = 1.0
        return target