# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 16:32:08 2018

@author: Michael
"""

#ipmorting the neccesary librariesk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras 
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

#importing the dataset
dataset=pd.read_csv('heart.csv')
train_data=dataset.drop(['target'],axis=1)
train_target=dataset['target']
train_data=StandardScaler().fit_transform(train_data)
x_train,x_test,y_train,y_test=train_test_split(train_data,train_target,random_state=7,test_size=0.2)
the_monitor=EarlyStopping(patience=5)

#build the neural network
model=Sequential()
model.add(Dense(16,activation='relu',input_shape=(train_data.shape[1],)))
model.add(Dropout(0.5))
model.add(Dense(8,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer=RMSprop(lr=0.01),loss='binary_crossentropy',metrics=['accuracy'])    
model_results=model.fit(x_train,y_train,epochs=53,validation_data=[x_test,y_test],verbose=True)

#the following variables will help follow values of loss and accuracy on both validation and training
model_history=model_results.history
mae=model_history['val_acc']
loss=model_results.history['loss']
val_loss=model_results.history['val_loss']
acc=model_results.history['acc']
val_acc=model_results.history['val_acc']








