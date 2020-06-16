# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

#img = cv2.imread('bananos.jpg')
#img = cv2.imread('fruta.jpg')
img = cv2.imread('000078.jpg')

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

I = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



umbral,_=cv2.threshold(I,0,255,cv2.THRESH_OTSU)




mascara=np.uint8((I<umbral)*255)

output=cv2.connectedComponentsWithStats(mascara,4,cv2.CV_32S)
cantObj=output[0]
labels=output[1]
stats=output[2]

mascara=(np.argmax(stats[:,4][1:])+1==labels)

mascara=ndimage.binary_fill_holes(mascara).astype(int)

mascara1=np.uint8(mascara*255)

_,contours,_=cv2.findContours(mascara1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

cnt=contours[0]

P=cv2.arcLength(cnt,True)
A=cv2.contourArea(cnt)

#CONVEX HULL

hull=cv2.convexHull(cnt)
puntosConvex=hull[:,0,:]
m,n=mascara1.shape
ar=np.zeros((m,n))
mascaraConvex=np.uint8(cv2.fillConvexPoly(ar,puntosConvex,1))


#Bounding box rotado

rect=cv2.minAreaRect(cnt)
box=np.int0(cv2.boxPoints(rect))

m,n=mascara1.shape
ar=np.zeros((m,n))
mascaraRect=np.uint8(cv2.fillConvexPoly(ar,box,1))


#Boundiung box recto

x,y,w,h=cv2.boundingRect(cnt)
cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)


_,contours,_=cv2.findContours(mascaraConvex,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img,contours,-1,(0,0,255),1)

_,contours,_=cv2.findContours(mascaraRect,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img,contours,-1,(255,0,0),1)


dato=I.flatten()

rojo=img[:,:,0].flatten()
verde=img[:,:,1].flatten()
azul=img[:,:,2].flatten()

plt.hist(rojo,bins=1000,histtype='stepfilled',color="red")
plt.hist(verde,bins=1000,histtype='stepfilled',color="green")
plt.hist(azul,bins=1000,histtype='stepfilled',color="blue")


segColor=np.zeros((m,n,3)).astype('uint8')
segColor[:,:,0]=np.uint8(img[:,:,0]*mascara)
segColor[:,:,1]=np.uint8(img[:,:,1]*mascara)
segColor[:,:,2]=np.uint8(img[:,:,2]*mascara)

segGrey=np.zeros((m,n))
segGrey[:,:]=np.uint8(I*mascara)


cv2.imshow('imagen',segColor)

#cv2.waitKey(0)
#cv2.destryAllWindows()