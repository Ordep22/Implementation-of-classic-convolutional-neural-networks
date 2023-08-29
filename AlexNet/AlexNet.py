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