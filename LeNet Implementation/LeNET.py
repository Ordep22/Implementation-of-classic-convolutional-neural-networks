import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import  Dense, Flatten, Conv2D, AveragePooling2D
from tensorflow.keras import datasets
from tensorflow.keras.utils import  to_categorical

class Data:

    def __init__(self):
        pass

    def Load(self):
        #Slit dataset beteween train and text sets:
        (xTrain,yTrain), (xTest,yTest) = datasets.fashion_mnist.load_data()


        #Check the shape of new data
        print('xTrain shape:', xTrain.shape)
        print(f"Train samples:{xTrain.shape[0]}")
        print(f"Test samples:{xTest.shape[0]}")
        print(f"Image samples:{xTrain[0].shape}")

        return (xTrain,yTrain), (xTest,yTest)

    def Process(self,xTrain,yTrain,xTest,yTest):

        self.xTrain = xTrain

        self.yTrain = yTrain

        self.xTest = xTest

        self.yTest = yTest

        #Add a new axis
        xTrain = xTrain[:,:,:,np.newaxis] #This meas tath the lis will be a new axises
        xTest = xTest[:,:,:,np.newaxis] #This meas tath the lis will be a new axises
        print('xTrain shape:', xTrain.shape)
        print(f"Train samples:{xTrain.shape[0]}")
        print(f"Test samples:{xTest.shape[0]}")
        print(f"Image samples:{xTrain[0].shape}")

        #Cover class vectors to binary class matrices.
        numClass  = 10
        #This function return a binary matrixes
        yTrain = to_categorical(yTrain,numClass)
        yTest = to_categorical(yTest,numClass)

        #Data normalization
        xTrain = xTrain.astype('float32')
        xTest = xTest.astype('float32')
        xTrain /= 255
        xTest /= 255


class LeNet(Sequential):
    def __init__(self, inputShape, nbClasses):

        super().__init__()

        self.add(Conv2D(6,kernelSize = (5,5),strides = (1,1),activation= 'tanh',
                        inputShape = inputShape,padding="same"))
        self.add(AveragePooling2D(pool_size=(2,2),strides=(2,2),padding= 'valid'))
        self.add(Conv2D(16,kernel_size=(5,5),strides=(2,2),padding= 'valid'))
        self.add(Flatten())
        self.add(Dense(120,activation= 'tanh'))
        self.add(Dense(84,activation= 'tanh'))
        self.add(Dense(nbClasses,activation= 'sotmax'))
        self.compile(optimizer='adam',loss = categorical_crossentropy,metrics=['accuracy'])





if __name__ == "__main__":

            Data = Data()

            LeNet = LeNet()

            (xTrain, yTrain), (xTest, yTest) =  Data.Load()

            Data.Process(xTrain, yTrain, xTest, yTest)

            #Return the elements that are will insert on the Class LeNet




















