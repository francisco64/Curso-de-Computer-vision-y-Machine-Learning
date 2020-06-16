#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 17:41:10 2020

@author: franciscorealescastro
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('cuadro.jpg')


imG=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


m,n,c=img.shape

pts_src = np.array([[42, 12], [193, 50],[40, 304], [198, 271]])


pts_dst = np.array([[0, 0], [n, 0],[0, m], [n, m]])


h, status = cv2.findHomography(pts_src, pts_dst)
#
im2 = cv2.warpPerspective(img, h, (n,m))

cv2.imshow('original',img)
## 
cv2.imshow('proyeccion',im2)
