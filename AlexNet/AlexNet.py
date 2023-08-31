import keras
#Sequential from keras.models, this gets ours neural networks as Sequential network
#As we know, it can be sequential layers or graph
from keras.models import Sequential

#Importing, Dense, Actiation, `Flatten, Activation, Dropout, Conv2D and Maxpooling
#Dropout is a technique used to prevet a model from overftting.
from keras.layers import  Dense, Activation, Dropout, Flatten, Conv2D, MaxPool2D

#For normalization
from keras.layers.normalization import  BatchNormalization

import numpy as np

image_shape  = (227,227,3)

#Instantiete on empyt model
np.random.seed(1000)

#It starts here
model  = Sequential()

#1st Convolutional Layer
#First layer has 96 filters, the input shape is 227x227x3
#Kernel Size id 11x11, String 4x4, relu is the activation function
model.add(Conv2D(filters = 96,input_shape=image_shape, kernel_size= (11,11), strides= (4,4), padding = 'valid'))
model.add(Activation('relu'))

#Max Pooling
model.add(MaxPool2D(pool_size=(3,3), strides=(2,2), padding  = 'valid'))

#2st Convolutional Layer
model.add(Conv2D(filters= 256, kernel_size=(5,5),strides= (1,1), padding='valid'))
model.add(Activation('relu'))

#Max Pooling
model.add(MaxPool2D(pool_size=(3,3),strides = (2,2),padding='valid'))

#3rd Convolutional Layer
model.add(Conv2D(filters = 384, kernel_size=(3,3), strides=(1,1),padding = 'valid'))
model.add(Activation('relu'))

#4rd Convolutional Layer
model.add(Conv2D(filters = 384, kernel_size=(3,3), strides=(1,1),padding = 'valid'))
model.add(Activation('relu'))

#5rd Convolutional Layer
model.add(Conv2D(filters = 256, kernel_size=(3,3), strides=(1,1),padding = 'valid'))
model.add(Activation('relu'))

#Max Pooling
model.add(MaxPool2D(pool_size=(3,3),strides = (2,2),padding='valid'))


#Passing it to a Fully Connected layer, Here we do flatten!
model.add(Flatten())

#1st Fully Connected Layer has 4096 neurons
model.add(Dense(4096,input_shape=(227*227*3,)))
model.add(Activation('relu'))

#Add Dropout to preventoverfitting
model.add(Dropout(0.4))

#2st Fully Connected Layer has 4096 neurons
model.add(Dense(4096))
model.add(Activation('relu'))
#Add Dropout
model.add(Dropout(0.4))

#Out layer
model.add(Dense(1000))
model.add(Activation('softmax'))

model.summary()

#Compile the model
model.compile(loss=keras.losses.categorical_crossentropy,optimizer='adam',metrics=["accuracy"])















