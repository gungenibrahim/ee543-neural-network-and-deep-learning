# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 16:51:12 2018

@author: gunge
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 11:49:19 2018

@author: gunge
"""

import keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, PReLU
from keras.optimizers import SGD, Adadelta, adam
import scipy.io as sio
import matplotlib.pyplot as plt



# Generate dummy data
import numpy as np
#x_train = np.random.random((1000, 20))
#y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
#x_test = np.random.random((100, 20))
#y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

Etrain_data = sio.loadmat('Etrain_extended.mat')
Etrain= Etrain_data['Etrain_extended']
Etrain_label_data = sio.loadmat('Etrain_extended_label.mat')
Etrain_label = Etrain_label_data['Etrain_extended_label']

Etest_data = sio.loadmat('Etest_extended.mat')
Etest = Etest_data['Etest_extended']
Etest_label_data = sio.loadmat('Etest_extended_label.mat')
Etest_label = Etest_label_data['Etest_extended_label']
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
#%92
#model.add(Dense(2048, activation='relu', input_dim=3750))
##model.add(Dropout(0.1))
##model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
#model.add(Dense(1024, activation='relu'))
##model.add(Dropout(0.1))
#model.add(Dense(256, activation='relu'))
##model.add(Dropout(0.1))
#model.add(Dense(64, activation='relu'))
##model.add(Dropout(0.1))
#model.add(Dense(16, activation='relu'))
##model.add(Dropout(0.5))
#model.add(Dense(3, activation='softmax')) 

model.add(Dense(1250, activation='relu', input_dim=3750))
#model.add(Dropout(0.1))
#model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
#model.add(Dense(1024, activation='relu'))
#model.add(Dropout(0.1))
model.add(Dense(250, activation='tanh'))
model.add(Dropout(0.01))
model.add(Dense(50, activation='tanh'))
model.add(Dropout(0.1))
model.add(Dense(10, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax')) 


keras.initializers.Ones()

sgd = SGD(lr=0.0001, decay=1e-4, momentum=0.9, nesterov=True)


model.compile(loss='squared_hinge',
              optimizer='sgd',
              metrics=['accuracy'])

history_ML = model.fit(x_train, y_train,validation_data=(x_test, y_test),epochs=400)

score_ML = model.evaluate(x_test, y_test, batch_size=10)


print('Test accuracy:', score_ML)


predictions_ML = model.predict(x_test)
plt.figure()
plt.plot(x_train[350])





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

















