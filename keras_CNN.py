# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 17:10:55 2018

@author: gunge
"""

from keras.datasets import mnist

import matplotlib.pyplot as plt
from keras.utils import to_categorical
import scipy.io as sio
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten




#download mnist data and split into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

Etrain_data = sio.loadmat('Etrain_3d.mat')
Etrain= Etrain_data['Etrain_3d']
Etrain_label_data = sio.loadmat('Etrain_label.mat')
Etrain_label = Etrain_label_data['Etrain_label']

Etest_data = sio.loadmat('Etest_3d.mat')
Etest = Etest_data['Etest_3d']
Etest_label_data = sio.loadmat('Etest_label.mat')
Etest_label = Etest_label_data['Etest_label']
#

class_names = ['Healty', 'Static_exan', 'Broken_mag' ]


X_train = Etrain
y_train = Etrain_label

X_test = Etest
y_test = Etest_label


#plot the first image in the dataset
plt.plot(X_train[0])

#X_train[0].shape

X_train = X_train.reshape(450,1250,3,1)
X_test = X_test.reshape(30,1250,3,1)


#one-hot encode target column
#y_train = to_categorical(y_train)
#y_test = to_categorical(y_test)
#y_train[0]



#create model
model = Sequential()
#add model layers
model.add(Conv2D(64, kernel_size=2, activation='relu', input_shape=(1250,3,1)))
model.add(Conv2D(32, kernel_size=2, activation='relu'))
model.add(Flatten())
model.add(Dense(3, activation='softmax'))


#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)

#predict first 4 images in the test set
model.predict(X_test[:4])

#
