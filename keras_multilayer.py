# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 11:49:19 2018

@author: gunge
"""

import keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import scipy.io as sio
import matplotlib.pyplot as plt



# Generate dummy data
import numpy as np
#x_train = np.random.random((1000, 20))
#y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
#x_test = np.random.random((100, 20))
#y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

Etrain_data = sio.loadmat('Etrain.mat')
Etrain= Etrain_data['Etrain']
Etrain_label_data = sio.loadmat('Etrain_label.mat')
Etrain_label = Etrain_label_data['Etrain_label']

Etest_data = sio.loadmat('Etest.mat')
Etest = Etest_data['Etest']
Etest_label_data = sio.loadmat('Etest_label.mat')
Etest_label = Etest_label_data['Etest_label']
#

class_names = ['Healty', 'Static_exan', 'Broken_mag' ]




x_train = Etrain
y_train = Etrain_label
x_test = Etest
y_test = Etest_label



model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(20, activation='relu', input_dim=3750))
model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)

score = model.evaluate(x_test, y_test, batch_size=128)


print('Test accuracy:', score)


predictions = model.predict(x_train)
plt.figure()
plt.plot(x_train[350])
print('sample accuracy:', predictions[350])




#
#
#def plot_image(i, predictions_array, true_label, img):
#  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
#  plt.grid(False)
#  plt.xticks([])
#  plt.yticks([])
#  
#  plt.plot(img)
#
#  predicted_label = np.argmax(predictions_array)
#  
#  print('predicted label:', predicted_label)
#  
##  if predicted_label == true_label:
##    color = 'blue'
##  else:
##    color = 'red'
#  
#  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
#                                100*np.max(predictions_array),
#                                class_names[true_label]),
#                                color=color)
#
#def plot_value_array(i, predictions_array, true_label):
#  predictions_array, true_label = predictions_array[i], true_label[i]
#  plt.grid(False)
#  plt.xticks([])
#  plt.yticks([])
#  thisplot = plt.bar(range(10), predictions_array, color="#777777")
#  plt.ylim([0, 1]) 
#  predicted_label = np.argmax(predictions_array)
# 
#  thisplot[predicted_label].set_color('red')
#  thisplot[true_label].set_color('blue')
#  
#
#
#i = 0
#plt.figure(figsize=(6,3))
#plt.subplot(1,2,1)
#plot_image(i, predictions, Etest_label, Etest)
##plt.subplot(1,2,2)
##plot_value_array(i, predictions,  Etest)
#
#

















