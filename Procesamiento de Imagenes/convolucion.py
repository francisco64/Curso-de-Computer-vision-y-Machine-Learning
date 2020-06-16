#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 10:22:36 2020

@author: franciscorealescastro
"""

import cv2
import numpy as np

img = cv2.imread('bananos.jpg')


cv2.imshow('Original',img)

kernel_3x3 = np.ones((3,3),np.float32) / 9.0
output=cv2.filter2D(img,-1,kernel_3x3) 
cv2.imshow('3x3',output)


kernel_5x5 = np.ones((5,5),np.float32) / 25.0
output=cv2.filter2D(img,-1,kernel_5x5)
cv2.imshow('5x5',output)


kernel_31x31 = np.ones((31,31),np.float32) / (31*31)
output=cv2.filter2D(img,-1,kernel_31x31)
cv2.imshow('31x31',output)


output=cv2.GaussianBlur(img,(3,3),0)
cv2.imshow('Gauss desv=3x3',output)

output=cv2.GaussianBlur(img,(11,11),0)

cv2.imshow('Gauss desv=11x11',output)

output=cv2.GaussianBlur(img,(21,21),0)

cv2.imshow('Gauss desv=21x21',output)