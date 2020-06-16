#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 23:02:39 2020

@author: franciscorealescastro
"""

import cv2
import numpy as np


a = np.zeros((50,100))
b = np.ones((50,100))

#img=np.uint8(255*np.concatenate((a,b),axis=0))
img=cv2.imread('bananos.jpg',0)

gx=cv2.Sobel(img,cv2.CV_64F,1,0,5)
gy=cv2.Sobel(img,cv2.CV_64F,0,1,5)
mag,ang=cv2.cartToPolar(gx,gy)

gx=cv2.convertScaleAbs(gx)
gy=cv2.convertScaleAbs(gy)
mag=cv2.convertScaleAbs(mag)
ang=(180/np.pi)*ang

imgFilt=cv2.GaussianBlur(img,(5,5),0)
lap=cv2.convertScaleAbs(cv2.Laplacian(imgFilt,cv2.CV_64F,5))

canny=cv2.Canny(img,25,150)


cv2.imshow('Gx',gx)
cv2.imshow('Gy',gy)
cv2.imshow('mag',mag)
cv2.imshow('Lap',lap)
cv2.imshow('Canny',canny)

