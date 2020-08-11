#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 16:36:42 2020

@author: franciscoreales
"""

import cv2


clas_caras=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
clas_ojos = cv2.CascadeClassifier('haarcascade_eye.xml')

cap=cv2.VideoCapture(0)

while(True):
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    caras=clas_caras.detectMultiScale(gray)
    
    for (x,y,w,h) in caras:
        img=cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=img[y:y+h,x:x+w]
        ojos=clas_ojos.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in ojos:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            
    cv2.imshow('img',img) 
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()