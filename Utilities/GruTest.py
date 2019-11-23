import keras
from keras import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Activation, GRU, TimeDistributed, Reshape

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, 28).astype('float32')
input_shape = (28, 28)                    
# Making sure that the values are float so that we can get decimal points after division
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])



model = Sequential()

#model.add(TimeDistributed(Conv2D(36, kernel_size=(5,5)),input_shape=input_shape))
#model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2))))

#model.add(TimeDistributed(Flatten()))

#model.add(CuDNNLSTM(128))

model.add(CuDNNGRU(128, input_shape = input_shape, return_sequences = True))
model.add(CuDNNGRU(128, return_sequences = False))
model.add(Activation('sigmoid'))
    
    
model.add(Dense(90))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

model.fit(x=x_train,
          y=y_train, 
          epochs=5,
          batch_size=256)

acc = model.evaluate(x_test, y_test)
print('acc test = ',acc[1],"%")

