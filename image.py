#!/usr/bin/python
# encoding: utf-8
import random
import os
import numpy as np

import cv2

def distort_image(im, hue, sat, val):
    im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2HSV_FULL)

    im[:, :, 0] = ((im[:, :, 0] + hue * 255) % 255).astype(np.uint8)
    im[:, :, 1] = np.clip(im[:, :, 1] * sat, 0, 255).astype(np.uint8)
    im[:, :, 2] = np.clip(im[:, :, 2] * val, 0, 255).astype(np.uint8)

    return cv2.cvtColor(im, cv2.COLOR_HSV2RGB_FULL)

def rand_scale(s):
    scale = random.uniform(1, s)
    if(random.randint(1,10000)%2): 
        return scale
    return 1./scale

def random_distort_image(im, hue, saturation, exposure):
    dhue = random.uniform(-hue, hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)
    return distort_image(im, dhue, dsat, dexp)

def data_augmentation(img, shape, jitter, hue, saturation, exposure):
    oh = img.shape[0]  
    ow = img.shape[1]
    
    dw =int(ow*jitter)
    dh =int(oh*jitter)

    pleft  = random.randint(-dw, dw)
    pright = random.randint(-dw, dw)
    ptop   = random.randint(-dh, dh)
    pbot   = random.randint(-dh, dh)

    swidth =  ow - pleft - pright
    sheight = oh - ptop - pbot

    sx = float(swidth)  / ow
    sy = float(sheight) / oh
    
    flip = random.randint(1,10000)%2
    pad = cv2.copyMakeBorder(img, dh, dh, dw, dw, cv2.BORDER_CONSTANT)
    cropped = pad[dh+ptop:dh+ptop + sheight - 1,dw+pleft:dw+pleft + swidth - 1,:]

    dx = (float(pleft)/ow)/sx
    dy = (float(ptop) /oh)/sy

    sized = cv2.resize(np.array(cropped), shape, interpolation=cv2.INTER_LINEAR)

    if flip: 
        sized = cv2.flip(sized, 1)
    img = random_distort_image(sized, hue, saturation, exposure)
    
    return img, flip, dx,dy,sx,sy 

def fill_truth_detection(labpath, w, h, flip, dx, dy, sx, sy):
    max_boxes = 50
    label = np.zeros((max_boxes,5))
    if os.path.getsize(labpath):
        bs = np.loadtxt(labpath)
        if bs is None:
            return label
        bs = np.reshape(bs, (-1, 5))
        cc = 0
        for i in range(bs.shape[0]):
            x1 = bs[i][1] - bs[i][3]/2
            y1 = bs[i][2] - bs[i][4]/2
            x2 = bs[i][1] + bs[i][3]/2
            y2 = bs[i][2] + bs[i][4]/2
            
            x1 = min(0.999, max(0, x1 * sx - dx)) 
            y1 = min(0.999, max(0, y1 * sy - dy)) 
            x2 = min(0.999, max(0, x2 * sx - dx))
            y2 = min(0.999, max(0, y2 * sy - dy))
            
            bs[i][1] = (x1 + x2)/2
            bs[i][2] = (y1 + y2)/2
            bs[i][3] = (x2 - x1)
            bs[i][4] = (y2 - y1)

            if flip:
                bs[i][1] =  0.999 - bs[i][1] 
            
            if bs[i][3] < 0.001 or bs[i][4] < 0.001:
                continue
            label[cc] = bs[i]
            cc += 1
            if cc >= 50:
                break

    label = np.reshape(label, (-1))
    return label

def load_data_detection(imgpath, shape, jitter, hue, saturation, exposure):
    labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png','.txt')

    ## data augmentation
    img = cv2.imread(imgpath.replace('.jpg', '.png'), cv2.IMREAD_UNCHANGED)
    img,flip,dx,dy,sx,sy = data_augmentation(img, shape, jitter, hue, saturation, exposure)
    label = fill_truth_detection(labpath, img.shape[1], img.shape[0], flip, dx, dy, 1./sx, 1./sy)
    return img,label
