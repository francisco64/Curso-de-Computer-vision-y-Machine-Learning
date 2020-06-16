#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 20:59:49 2020

@author: franciscorealescastro
"""

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.manifold import TSNE

def get_hog():
    winSize = (20,20)
    blockSize=(8,8)
    blockStride = (4,4)
    cellSize=(8,8)
    nbins = 9
    derivAperture = 1
    winSigma = 2.
    histrogramType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlavels = 64
    signedGradient = True 
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histrogramType,L2HysThreshold,gammaCorrection,nlavels,signedGradient)
    return hog

def escalar(img,m,n):
    if m>n:
        imgN=np.uint8(255*np.ones((m,round((m-n)/2),3)))
        escalada=np.concatenate((np.concatenate((imgN,img),axis=1),imgN), axis=1)

    else:
        imgN=np.uint8(255*np.ones((round((n-m)/2),n,3)))
        escalada=np.concatenate((np.concatenate((imgN,img),axis=0),imgN), axis=0)
    
    img = cv2.resize(escalada, (20,20))
    
    return img
        
def obtenerDatos():
    posiblesEtiq=['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z']
    
    datos = []
    etiquetas = []
    
    for i in range(1,26):
        for j in posiblesEtiq:
            img=cv2.imread(j+'-'+str(i)+".jpg")
            if img is not None:
                m,n,_=img.shape
                if m !=20 or n !=20:
                    img =escalar(img,m,n)
                etiquetas.append(np.where(np.array(posiblesEtiq)==j)[0][0])
                hog = get_hog()
                datos.append(np.array(hog.compute(img)))
    datos=np.array(datos)[:,:,0]
    etiquetas=np.array(etiquetas)
    return datos, etiquetas

def clasificadorCaracteres():
    datos, etiquetas=obtenerDatos()
    knn=KNeighborsClassifier(n_neighbors=1)
    knn.fit(datos,etiquetas)
    SVM=svm.SVC(kernel='linear', probability=True, random_state=0,gamma='auto')
    SVM.fit(datos,etiquetas)
    return knn, SVM
    

# datos, etiquetas = obtenerDatos()
# #-------------------------- Evaluacion KNN
# X_train,X_test,y_train,y_test=train_test_split(datos,etiquetas,test_size=0.2, random_state=np.random)
# knn=KNeighborsClassifier(n_neighbors=1)
# knn.fit(X_train,y_train)
# errorEntrenamientoKnn=(1-knn.score(X_train,y_train))*100
# print("Error de entrenamiento del Knn es: "+str(round(errorEntrenamientoKnn,2))+"%")
# errorPruebaKnn=(1-knn.score(X_test,y_test))*100
# print("Error de prueba del Knn es: "+str(round(errorPruebaKnn,2))+"%")

# prediccionKnn=knn.predict(X_test)
# errorKnn=100*(1-cross_val_score(knn,datos,etiquetas,cv=10))

# print("Knn cross val: "+str(round(errorKnn.mean(),2))+"+-"+str(round(errorKnn.std(),2)))
# #------------------------Matriz de confusion
# # plt.imshow(confusion_matrix(y_test,prediccionKnn), interpolation = "nearest")

# # plt.title("Matriz de cofusion Knn")
# # plt.xlabel("Prediccion")
# # plt.ylabel("verdadera etiqueta")
# #-------------------------SVM
# SVM=svm.SVC(kernel='linear', probability=True, random_state=0,gamma='auto')

# SVM.fit(X_train,y_train)
# errorSVM=100*(1-cross_val_score(SVM,datos,etiquetas,cv=10))
# print("SVM cross val: "+str(round(errorSVM.mean(),2))+"+-"+str(round(errorSVM.std(),2)))

# #-----------------------T-SNE

# X=TSNE(n_components=2).fit_transform(datos)
# posiblesEtiq=['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z']
# x_min, x_max=np.min(X,0),np.max(X,0)
# X=(X-x_min)/(x_max-x_min)

# for i in range(0,len(X)):
#     plt.text(X[i,0],X[i,1],str(posiblesEtiq[etiquetas[i]]), color = plt.cm.Set1(3*float(etiquetas[i])/99))