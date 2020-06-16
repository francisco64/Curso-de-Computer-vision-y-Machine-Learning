#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 10:29:22 2020

@author: franciscorealescastro
"""
import cv2
import numpy as np

img=cv2.imread('iphone.jpg')
imG=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

canny=cv2.Canny(imG,25,150)

kernel = np.ones((5,5), np.uint8) 
bordes = cv2.dilate(canny, kernel) 

_,contours,_ = cv2.findContours(bordes,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

objetos=bordes.copy()
cv2.drawContours(objetos, [max(contours, key = cv2.contourArea)], -1, 255, thickness=-1)
objetos=objetos/255

seg=np.ones(img.shape)
seg[:,:,0]=objetos*img[:,:,0]+255*(objetos==0)
seg[:,:,1]=objetos*img[:,:,1]+255*(objetos==0)
seg[:,:,2]=objetos*img[:,:,2]+255*(objetos==0)
seg=np.uint8(seg)

cv2.imshow('original',img)
cv2.imshow('segmentada',seg)