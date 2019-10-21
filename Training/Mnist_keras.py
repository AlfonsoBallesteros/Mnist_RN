#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install keras')


# In[2]:


from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[3]:


# save model and architecture to single file
model.save("model.h5")
print("Saved model to disk")


# In[5]:


# load and evaluate a saved model
#from numpy import loadtxt
from keras.models import load_model
 
# load model
model = load_model('../Models/model.h5')
# summarize model.
model.summary()
# load datasetdataset = loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables = dataset[:,0:8] = dataset[:,8]
# evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))


# In[6]:


from matplotlib import pyplot as plt


# In[7]:


x_test[0]


# In[10]:


import numpy as np
img = np.reshape(x_test[0], (28,28))
x_test[0].shape
plt.imshow(img)
plt.show()


# In[17]:


img1 = np.reshape(x_test[1], (1,28,28,1))
pred = model.predict(img1)
print(pred)


# In[18]:


result = np.argmax(pred)
print(result)


# In[23]:


get_ipython().system('pip install pillow')


# In[24]:


ejemplo = plt.imread('../Image/example.png')


# In[35]:


plt.imshow(ejemplo)
plt.show()
print(ejemplo.shape)
grises = ejemplo[:,:,0]
plt.imshow(grises)
plt.show()


# In[36]:


img2 = np.reshape(grises, (1,28,28,1))
pred1 = model.predict(img2)
print(pred)
result1 = np.argmax(pred)
print(result)


# In[42]:


tres = plt.imread('../Image/tres.png')
plt.imshow(tres)
plt.show()
print(tres.shape)
gris = tres[:,:,0]
plt.imshow(gris)
plt.show()


# In[43]:


img3 = np.reshape(gris, (1,28,28,1))
pred2 = model.predict(img3)
print(pred2)
result2 = np.argmax(pred2)
print(result2)


# In[44]:


from keras.models import load_model
import numpy as np
from matplotlib import pyplot as plt

# load model
model = load_model('../Models/model.h5')

imagen = plt.imread('../Image/tres.png')
plt.imshow(imagen)
plt.show()
print(imagen.shape)
escala_grises = imagen[:,:,0]
plt.imshow(escala_grises)
plt.show()
rezise = np.reshape(escala_grises, (1,28,28,1))
predic = model.predict(rezise)
print(predic)
resultado = np.argmax(predic)
print(resultado)


# In[ ]:




