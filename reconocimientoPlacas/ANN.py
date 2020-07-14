#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 21:55:26 2020

@author: franciscorealescastro
"""

from reconocimientoCaracteres import obtenerDatos
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

def modelo1():
    model=Sequential()
    model.add(Dense(200, input_dim=144))
    model.add(Dense(180))
    model.add(Dense(150))
    model.add(Dense(34, activation='softmax'))   
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
    return model
model=modelo1()
datos,etiquetas =obtenerDatos()
clases=34
X_train,X_test,y_train,y_test=train_test_split(datos,etiquetas,test_size=0.2,random_state=np.random)
y_trainOneHot=tf.one_hot(y_train,clases)
y_testOneHot=tf.one_hot(y_test,clases)
model.fit(X_train,y_trainOneHot,epochs=100,batch_size=100)
prediccion=model.predict(X_test)
y_pred=np.argmax(prediccion,1)
errorPrueba=100*(1-np.sum(y_pred==y_test)/len(y_test))
print("El error de prueba es: "+str(round(errorPrueba,2)))