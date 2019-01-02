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


model.add(Dense(1250, activation='relu', input_dim=3750))
#model.add(Dropout(0.1))
#model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
#model.add(Dense(1024, activation='relu'))
#model.add(Dropout(0.1))
model.add(Dense(250, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='tanh'))
model.add(Dropout(0.5))
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








#plt.plot(history_ML.history['acc'])
#plt.plot(history_ML.history['val_acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()
## summarize history for loss
#plt.plot(history_ML.history['loss'])
#plt.plot(history_ML.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()
##
#plt.plot(X_test[444])
#plt.ylabel('current')
#plt.title('test data test(444)')
#plt.show()
#
#plt.plot(x_test[444])
#plt.ylabel('current')
#plt.title('test data test(444)')
#plt.show()
#
#pred = predictions_ML[444]
#actual = Etest_label[444]
# 
## create plot
#fig, ax = plt.subplots()
#index = np.arange(len(class_names))
#bar_width = 0.35
#opacity = 0.8
# 
#rects1 = plt.bar(index, pred, bar_width,
#                 alpha=opacity,
#                 color='b',
#                 label='Prediction')
# 
#rects2 = plt.bar(index + bar_width, actual, bar_width,
#                 alpha=opacity,
#                 color='g',
#                 label='Actual')
# 
#plt.xlabel('faults')
#plt.ylabel('Scores')
#plt.title('Prediction and Actual class at test data(444)')
#plt.xticks(index + bar_width, class_names)
#plt.legend()
# 
#plt.tight_layout()
#plt.show()


















