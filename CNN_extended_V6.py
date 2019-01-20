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
from keras.layers import Dense, Conv2D, Flatten, Conv1D, MaxPooling1D, Dropout

import numpy as np

from sklearn import preprocessing


#download mnist data and split into train and test sets
#(X_train, y_train), (X_test, y_test) = mnist.load_data()

Etrain_data = sio.loadmat('Etrain_extended_3d.mat')
Etrain= Etrain_data['Etrain_extended_3d']
Etrain_label_data = sio.loadmat('Etrain_extended_label.mat')
Etrain_label = Etrain_label_data['Etrain_extended_label']

Etest_data = sio.loadmat('Etest_extended_3d.mat')
Etest = Etest_data['Etest_extended_3d']
Etest_label_data = sio.loadmat('Etest_extended_label.mat')
Etest_label = Etest_label_data['Etest_extended_label']
#




class_names = ['Healty', 'Static_exan', 'Broken_mag' ]

X_train= Etrain
y_train = Etrain_label

X_test = Etest
y_test = Etest_label

X_train_fft = Etrain
X_test_fft = Etest


for i in range(0, 6750):
    for j in range(0,3):
        rms = np.sqrt(np.mean(X_train[i,:,j]**2))
        #X_train[i,:,j] = preprocessing.normalize([X_train[i,:,j]])
        X_train[i,:,j] = X_train[i,:,j]/rms 
        X_train_fourier = np.fft.fft(X_train[i,:,j])
        X_train_fft[i,:,j] = np.real(X_train_fourier)
        X_train_fft[i,:,j] = np.clip(X_train_fft[i,:,j],-30,30)
    
for i in range(0, 450):
    for j in range(0,3):
        rms = np.sqrt(np.mean(X_test[i,:,j]**2))
        #X_test[i,:,j] = preprocessing.normalize([X_test[i,:,j]])
        X_test[i,:,j] = X_test[i,:,j]/rms
        X_test_fourier = np.fft.fft(X_test[i,:,j])
        X_test_fft[i,:,j] = np.real(X_test_fourier)
        X_test_fft[i,:,j] = np.clip(X_test_fft[i,:,j],-30,30)

#plot the first image in the dataset
plt.plot(X_train[350])

#X_train[0].shape

#X_train = X_train.reshape(6750,1250,3,1)
#X_test = X_test.reshape(450,1250,3,1)


#one-hot encode target column
#y_train = to_categorical(y_train)
#y_test = to_categorical(y_test)
#y_train[0]



















#create model
model_cnn_fft = Sequential()
#add model layers
model_cnn_fft.add(Conv1D(2, kernel_size=3, activation='relu', input_shape=(1250,3)))
model_cnn_fft.add(MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last'))
model_cnn_fft.add(Conv1D(64, kernel_size=3, activation='relu'))
model_cnn_fft.add(MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last'))
model_cnn_fft.add(Dropout(0.1))
model_cnn_fft.add(Conv1D(128, kernel_size=2, activation='relu'))
model_cnn_fft.add(MaxPooling1D(pool_size=8, strides=None, padding='valid', data_format='channels_last'))
#model.add(Conv1D(8, kernel_size=2, activation='relu'))
#model.add(Conv1D(16, kernel_size=2, activation='relu'))

model_cnn_fft.add(Flatten())
model_cnn_fft.add(Dense(300, activation='relu'))
model_cnn_fft.add(Dropout(0.1))
model_cnn_fft.add(Dense(30, activation='relu'))
model_cnn_fft.add(Dropout(0.1))
model_cnn_fft.add(Dense(3, activation='softmax'))


#compile model using accuracy to measure model performance
model_cnn_fft.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#train the model
history_fft = model_cnn_fft.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=250 )




score = model_cnn_fft.evaluate(X_test, y_test, batch_size=3)


print('Test accuracy:', score)



#predict first 4 images in the test set
predictions_fft = model_cnn_fft.predict(X_test)
plt.figure()
#plt.plot(X_train[350])
print('sample accuracy:', predictions_fft[350])





print(history_fft.history.keys())
# summarize history for accuracy
plt.plot(history_fft.history['acc'])
plt.plot(history_fft.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history_fft.history['loss'])
plt.plot(history_fft.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()






#
#
#
#
##create model
#model_cnn = Sequential()
##add model layers
#model_cnn.add(Conv1D(2, kernel_size=3, activation='relu', input_shape=(1250,3)))
#model_cnn.add(MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last'))
#model_cnn.add(Conv1D(64, kernel_size=3, activation='relu'))
#model_cnn.add(MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last'))
#model_cnn.add(Dropout(0.1))
#model_cnn.add(Conv1D(128, kernel_size=2, activation='relu'))
#model_cnn.add(MaxPooling1D(pool_size=8, strides=None, padding='valid', data_format='channels_last'))
##model.add(Conv1D(8, kernel_size=2, activation='relu'))
##model.add(Conv1D(16, kernel_size=2, activation='relu'))
#
#model_cnn.add(Flatten())
#model_cnn.add(Dense(300, activation='relu'))
#model_cnn.add(Dropout(0.1))
#model_cnn.add(Dense(30, activation='relu'))
#model_cnn.add(Dropout(0.1))
#model_cnn.add(Dense(3, activation='softmax'))
#
#
##compile model using accuracy to measure model performance
#model_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
##train the model
#history = model_cnn.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=250 )
#
#
#
#
#score = model_cnn.evaluate(X_test, y_test, batch_size=3)
#
#
#print('Test accuracy:', score)
#
#
#
##predict first 4 images in the test set
#predictions = model_cnn.predict(X_test)
#plt.figure()
##plt.plot(X_train[350])
#print('sample accuracy:', predictions[350])
#

#
# list all data in history
#print(history.history.keys())
## summarize history for accuracy
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()
## summarize history for loss
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()
#













