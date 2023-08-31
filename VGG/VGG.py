#Sequential from keras.models, this gets our neral networks as Sequential noetworks.
#As we know, it can be sequential layers or graphs
import keras,os
from keras.models import  Sequential
import tensorflow as tf

#We are working with images. All the images are basically 2D
#One can go with the 3D if working with videos.
from keras.layers import Conv2D

#Avarege Pooling, Sum Poolong and Max Pooling are there.
#We choose Max pooling. Re collect all what I tought you. From Keras. Layers import
from keras.layers import MaxPool2D

#Well, we must flatten. It is the process of conversion all the resultant 2D arrays single long continuos liner vector.
#This is mandatory, folks
from keras.layers import Flatten

#This is the last step! Yes, full connection of the neral network is performed with thei Dense.
from keras.layers import Dense

#We are goig to use ImageDataGenerator from Keras and hence import it as well! It helps in rescale, rotate, zoom, flip etc.
from keras.preprocessing.image import  ImageDataGenerator

from matplotlib import  pyplot as plt

import numpy as np

mnist  = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

rows, cols  = 28,28

x_train = x_train.reshape(x_train.shape[0],rows,cols,1)
x_test = x_test.reshape(x_test.shape[0],rows,cols,1)

input_shape = (rows, cols, 1)

#Normalize datas
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train  = x_train  /255.0
x_test = x_test /255.0

#One- hot encde the labels
y_train = tf.keras.utils.to_categorical(y_train,10)
y_test = tf.keras.utils.to_categorical(y_test,10)


def VggNet(input_shape):

    #Can we initialize the CNN and start the real coding?
    model = Sequential()


    model.add(Conv2D(input_shape= input_shape,filters=64,kernel_size=(3,3),padding="same",activation= "relu"))
    model.add(Conv2D(filters=64, kernel_size=(3,3),padding="same",activation= "relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

    #Flow the same procedure
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    #Flow the same procedure
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # Flow the same procedure
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # Flow the same procedure
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    #Here, what we are basically doing here is taking the 2-D array,
    #i.e pooled image pixels and converting them to a one dimensional single vector.
    model.load_weights('vggweights.h5')
    for layer in model.layers:
        layer.trainable = False
    model.add(Flatten())

    model.add(Dense(units=256,activation="relu"))
    model.add(Dense(units=256, activation="relu"))
    model.add(Dense(units=2, activation="softmax"))

    model.compile(optimizer='adam',loss = keras.losses.categorical_crossentropy,metrics=['accuracy'])

    return model


#We build it!
vggNet  = VggNet(input_shape)

#Number of epochs
epochs  = 10

#Can we train the model
history  = vggNet.fit(x_train,y_train,steps_per_epoch=100,epochs = epochs, batch_size= 128, verbose=1)

loss, acc  = vggNet.evaluate(x_test, y_test)

#Here is the most important thing to be learnet!
#Epochs - what is it? Simple, epoch is once all the images are processes one time individually
#Both forward and backward to the network.
#Epoch number can be determined by the trail and error.

print("Accuracy: ", acc)


#Trasnformation/ Reshape into 28x28 pixel
x_train = x_train.reshape(x_train.shape[0],28,28)
print("Trainnig Data",x_train.shape,y_train.shape)

x_test = x_test.reshape(x_test.shape[0],28,28)
print("Test Data", x_test.shape,y_test.shape)

#To visualize a simgle image at the index
imge_index  = 4444
plt.imshow(x_test[imge_index].reshape(28,28),cmap = 'Greys')

#To predict the output using the lenet model built
pred  = vggNet.predict(x_test[imge_index].reshape(1,rows,cols, 1))
print(pred.argmax())










