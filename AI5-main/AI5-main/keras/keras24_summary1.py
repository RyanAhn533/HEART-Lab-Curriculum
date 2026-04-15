from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#2. 모델

model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))

model.summary()
"""
dense (Dense)               (None, 3)                 6
dense_1 (Dense)             (None, 4)                 16
dense_2 (Dense)             (None, 3)                 15
dense_3 (Dense)             (None, 1)                 4
y = wx + b
  x'1  x'2
x1 x2 x3
"""