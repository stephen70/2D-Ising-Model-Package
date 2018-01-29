import numpy as np
import math
import copy
import keras
from keras import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Convolution1D
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
import pandas as pd
import tensorflow as tf
pd.set_option('display.width', 300)

df = pd.read_csv('training1.csv', header=None)

half = int(len(df.columns) / 2)
order = int(np.sqrt(half))

inp = df.iloc[:, :half]
out = df.iloc[:, half:]

inp2 = []
out2 = []

for i in np.arange(len(df)):
    inp2.append(np.array(inp.iloc[i]).reshape(4,4))
    out2.append(np.array(out.iloc[i]).reshape(4,4))

inp2 = np.array(inp2)
out2 = np.array(out2)
inp2 += 1
inp2 = inp2/2
out2 += 1
out2 = out2/2

inp2 = inp2.reshape(5,4,4,1)
out2 = out2.reshape(5,4,4,1)
print(inp2.shape)
print(inp2.ndim)

#
# pan = pd.Panel(inp2)
# inp = pan.swapaxes(1, 0).to_frame()
# pan = pd.Panel(out2)
# out = pan.swapaxes(1, 0).to_frame()

model = Sequential()
model.add(Convolution2D(32, (2,2), input_shape=(4,4,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), input_shape=(32,2,2,1)))

# model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
# model.add(Dense(16))
# model.add(Activation('sigmoid'))


model.compile(loss='mean_squared_error',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(inp2, out2)