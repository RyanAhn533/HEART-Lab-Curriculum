import numpy as np
from tensorflow.keras.datasets import mnist
import pandas as pd
(x_train, y_train), (x_test, y_test) = mnist.load_data()


np.set_printoptions(edgeitems=30, linewidth = 1024)
# print(x_train) #->0이 존내많이 나옴
"""
[[[0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  ...
 ...
  ...
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]]

 [[0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  ...
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]]]
"""
print(x_train[0], y_train[0])
print(x_train.shape, y_train.shape) #(60000, 28, 28) (10000,) / 가로 28 세로 28 짜리가 6만장 / 10000 <- 이게 1이면 흑백
print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,)

#y값을 구하는 법 np.unique
print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949], dtype=int64))
print(pd.value_counts(y_test))
#convolution 3 차원 
#cnn 4차원
import matplotlib.pyplot as plt
plt.imshow(x_train[3], 'tab20b')
plt.show()

