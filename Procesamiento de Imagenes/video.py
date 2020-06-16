#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 18:43:29 2020

@author: franciscorealescastro
"""
import numpy as np
import cv2
from scipy import ndimage


cap = cv2.VideoCapture('trafico6.mp4')
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video=[]
ret, frame1 = cap.read()
for i in range(0,length):

    ret, frame = cap.read()
    if ret:
        video.append(frame)
          
suma=np.zeros(frame1.shape)

for frameI in video:
    suma=suma+frameI
    
modeloFondo=np.uint8(suma/len(video))

for frameI in video:  
    #frameI=video[50]
    diferencia=cv2.absdiff(frameI,modeloFondo).astype('uint8')
    
    diferencia = cv2.GaussianBlur(diferencia,(11,11),0)
    diferencia=cv2.cvtColor(diferencia, cv2.COLOR_BGR2GRAY)
    

    u,_=cv2.threshold(diferencia,0,255,cv2.THRESH_OTSU)
    
    umbral=u
    #umbral=42
    #umbral=u
    
    bina=255*np.uint8(diferencia>1*umbral)

    
    bina=255*np.uint8(ndimage.binary_fill_holes(bina))
    
    _,contours,_ = cv2.findContours(bina, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    for i in contours:
        
        rect = cv2.boundingRect(i)
        x,y,w,h = rect
        if cv2.contourArea(i)>100:
            cv2.rectangle(frameI,(x,y),(x+w,y+h),(0,255,0),2)
    #break
    cv2.imshow('objetos',frameI)
    cv2.waitKey(500)
