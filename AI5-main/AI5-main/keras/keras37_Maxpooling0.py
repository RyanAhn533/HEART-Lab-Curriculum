#35_2에서 가져옴
# x_train, x_test는 reshape
# y_tset, y_train OneHotEncoding

import numpy as np
from tensorflow.keras.datasets import mnist
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from sklearn.metrics import accuracy_score

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data() #알아서 데이터 나눠줌
# print(x_train)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
 

print(x_train[0])
print("y_train[0] : ", y_train[0])

print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,) #색상 값이 숨겨져있다. 사실은 60000, 28, 28, 1
print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,) 

#2 모델구성

model = Sequential()
model.add(Conv2D(10, (3,3), input_shape=(28, 28, 1),
                 strides=1,
                #  padding='same'
                 ))  #26, 26, 10
model.add(MaxPooling2D()) #13, 13, 10
model.add(Conv2D(filters=9, kernel_size=(3,3), 
                 strides=1, padding='valid')) #11 , 11, 9
model.add(Conv2D(8, (2,2)))#10, 10, 8
# model.add(Flatten()) 
# model.add(Dense(units=8)) 
# model.add(Dense(units=9, input_shape=(8,))) 
# model.add(Dense(10, activation='softmax')) 




model.summary() #summary가 됨. 

'''
#3 컴파일, 훈련
model.compile( loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs=2, batch_size=128)

#4 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('로스 값', loss)
print('정확도 ', acc)
'''