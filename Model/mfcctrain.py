import tensorflow as tf
from tensorflow import keras
import numpy as np

train_data = np.load("../Data/Optimized/mfcc.npy")

train_labels = []

for i in range(470):
  train_labels.append([1,0,0])
  
for i in range(470):
  train_labels.append([0,1,0])

for i in range(470):
  train_labels.append([0,0,1])

train_labels = np.array(train_labels)


model = keras.Sequential([
      keras.layers.Dense(16796, input_shape=(16796,)),
      keras.layers.Dense(5589, activation='relu'),
      keras.layers.Dense(1863, activation='relu'),
      keras.layers.Dense(621, activation='relu'),
      keras.layers.Dense(207, activation='relu'),
      keras.layers.Dense(69, activation='relu'),
      keras.layers.Dense(23, activation='relu'),
      keras.layers.Dense(3, activation='softmax'),
      
])

model.compile(optimizer=keras.optimizers.SGD(lr = 0.01), loss="categorical_crossentropy", metrics = ["accuracy"])
model.fit(train_data, train_labels, batch_size = 32, epochs=2)

model.save("./mfcc/")