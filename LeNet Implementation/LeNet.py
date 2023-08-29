import matplotlib.pyplot as plt
import tensorflow as tf
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

def build_lenet(input_shape):

    #Sequential API
    model  = tf.keras.Sequential()

    #Convolution #1. Filters as we know, is 6. Filter size is 5x5, tanh is the activation function. 28x28 is the dimension.
    model.add(tf.keras.layers.Conv2D(filters=6,kernel_size=(5,5), strides=(1,1), activation= 'tanh',input_shape = input_shape))

    #SubSampling #1. Input  = 28x28x6. Output  = 14x14x6. Subsampling is simply avarage so we use avg_pool
    model. add(tf.keras.layers.AveragePooling2D(pool_size=(2,2),strides=(2,2)))

    #Convolution #2. Input  = 14x14x6. Output  = 10x10x6 conv2d
    model.add(tf.keras.layers.Conv2D(filters = 16, kernel_size=(5,5),strides=(1,1),activation='tanh'))

    # SubSampling #2. Input  = 28x28x6. Output  = 14x14x6. Subsampling is simply avarage so we use avg_pool
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

    # We must flatten for the future steps to happen.
    # It is the process of converting all the results 2D arrays as single long continuos linear vector
    model.add(tf.keras.layers.Flatten())

    #Fully connected #1. Input = 5x5x16. Output = 120
    model.add(tf.keras.layers.Dense(units=100, activation='tanh'))

    #Flattening here. It is the process of converting all the resultant 2D arrays as single long contiouus linear vector
    model.add(tf.keras.layers.Flatten())

    #Fully connected #2. Input = 120. Output = 84
    model.add(tf.keras.layers.Dense(units = 84, activation = 'tanh'))

    #output layer
    model.add(tf.keras.layers.Dense(units= 10, activation='softmax'))

    #Arguments passed like the past, nothing to worry!!
    model.compile(loss= 'categorical_crossentropy',optimizer = tf.keras.optimizers.SGD(lr = 0.1,momentum=0.0,decay=0.0),metrics=['accuracy'])

    return model

#We build it!
lenet  = build_lenet(input_shape)

#Number of epochs
epochs  = 10

#Can we train the model
history  = lenet.fit(x_train,y_train,epochs = epochs, batch_size= 128, verbose=1)

loss, acc  = lenet.evaluate(x_test, y_test)

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
pred  = lenet.predict(x_test[imge_index].reshape(1,rows,cols, 1))
print(pred.argmax())






























