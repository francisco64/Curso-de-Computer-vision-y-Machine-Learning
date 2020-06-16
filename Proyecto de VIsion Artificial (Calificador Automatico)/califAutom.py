#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 11:32:51 2020

@author: franciscorealescastro
"""

import cv2
import numpy as np



def obtenerRespuestas(img):
    canny=cv2.Canny(img,20,150)
    
    kernel= np.ones((5,5),np.uint8)
    
    bordes=cv2.dilate(canny,kernel)
    
    _,contour,_=cv2.findContours(bordes,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    objetos=bordes.copy()
    cv2.drawContours(objetos,[max(contour, key=cv2.contourArea)],-1,255,thickness=-1)
    
    output = cv2.connectedComponentsWithStats(objetos,4,cv2.CV_32S)
    numObj=output[0]
    labels=output[1]
    stats=output[2]
    
    mascara=np.uint8(255*(np.argmax(stats[:,4][1:])+1==labels))
    
    _,contours,_=cv2.findContours(mascara,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    cnt=contours[0]
    
    hull=cv2.convexHull(cnt)
    puntosConvex=hull[:,0,:]
    m,n=mascara.shape
    ar=np.zeros((m,n))
    mascaraConvex=np.uint8(cv2.fillConvexPoly(ar,puntosConvex,1))
    
    
    vertices=cv2.goodFeaturesToTrack(mascaraConvex,4,0.01,20)
    
    x=vertices[:,0,0]
    y=vertices[:,0,1]
    
    vertices=vertices[:,0,:]
    
    xo=np.sort(x)
    yo=np.sort(y)
    
    xn=np.zeros((1,4))
    yn=np.zeros((1,4))
    
    xn=(x==xo[2])*n+(x==xo[3])*n
    yn=(y==yo[2])*m+(y==yo[3])*m
    
    
    verticesN=np.zeros((4,2))
    verticesN[:,0]=xn
    verticesN[:,1]=yn
    
    vertices=np.int64(vertices)
    verticesN=np.int64(verticesN)
    
    h,_=cv2.findHomography(vertices,verticesN)
    
    im2=cv2.warpPerspective(img,h,(n,m))
    roi=im2[:,np.uint64(0.25*n):np.uint64(0.84*n)]
    
    opciones=['A','B','C','D','E','x']
    respuestas=[]
    
    for i in range(0,26):
        pregunta=roi[np.uint64(i*(m/26)):np.uint64((i+1)*(m/26)),:]
        sumI=[]
        for j in range(0,5):
            _,col=pregunta.shape
            sumI.append(np.sum(pregunta[:,np.uint64(j*(col/5)):np.uint64((j+1)*(col/5))])) 
        vmin=np.ones((1,5))*np.min(sumI)
    
        if np.linalg.norm(sumI-vmin)>0.17*col*m:
            sumI.append(float('inf'))
        else:sumI.append(-1)
        
        respuestas.append(opciones[np.argmin(sumI)])
    
    return respuestas

img=cv2.imread('foto_formato2.jpg',0)

respuestasFormato=np.array(obtenerRespuestas(img))

respuestasCorrectas=np.array(['B','C','D','E','B','C','C','A','A','B','C','A','E','A','A','A','B','C','A','A','B','C','A','A','B','C'])


calificacion=10*np.sum(respuestasCorrectas==respuestasFormato)/26