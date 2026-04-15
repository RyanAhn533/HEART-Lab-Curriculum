from __future__ import absolute_import, division, print_function, unicode_literals
from __future__ import annotations
####################
import numpy as np
import tensorflow
from tensorflow.keras.datasets import mnist
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping
##############
import warnings
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
###############
from sklearn.metrics import r2_score, accuracy_score

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000, 28, 28, 1)


y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)


print(x_train.shape, y_train.shape) #(60000, 28, 28) (10000,) / 가로 28 세로 28 짜리가 6만장 / 10000 <- 이게 1이면 흑백
print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,)


#데이터를 4차원으로 바꿔줘야함

#2. 모델
model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(28, 28, 1))) #27 27 10

# shape = (batch_size, rows, columns, channels)
# shape = (batch_size, heights, widths, channels) - 
# 6만의 데이터를 한번에 넣을 수 없으니 32개씩 잡아서 훈련시킨다.

# input shape  
#10 ->filter  (2,2) -> kernal_size
#Cnn 항상 4차원 shape 항상 3차원 

model.add(Conv2D(filters=20, kernel_size=(3,3))) 
model.add(Conv2D(15, (4,4)))

model.add(Flatten())
#PFM
model.add(Dense(units=9))
model.add(Dropout(0.2))
model.add(Dense(units=9, input_shape=(8,)))
#shape = (batch_size, input_dim)
model.add(Dense(10, activation='softmax'))
model.summary()



# 컴파일 및 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
import time 
start = time.time()
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=1, restore_best_weights=True)
end = time.time()
# 모델 체크포인트 파일 경로 설정
filepath = './save/mcp2/keras35/keras35_cnn4.hdf5'
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

# 모델 훈련
start = time.time()
model.fit(x_train, y_train, epochs=10, batch_size=1024, verbose=1, validation_split=0.1, callbacks=[es, mcp])
end = time.time()

# 모델 평가 및 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss)

y_predict = model.predict(x_test)

acc = accuracy_score(y_test, np.round(y_predict))
print("걸린시간은? : ", end-start, "초")
print('acc?', acc)
#걸린시간은? gpu on 4.74 초
#걸린시간은? gpu off 1.62 초