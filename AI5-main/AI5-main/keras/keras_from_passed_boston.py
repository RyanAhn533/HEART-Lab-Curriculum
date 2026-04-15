# keras25_input_shape.py copy

import numpy as np
import sklearn as sk
print(sk.__version__)   # 0.24.2
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time

#1. 데이터
dataset = load_boston()
print(dataset)
print(dataset.DESCR)   # DESCR = pandas 의 describe
print(dataset.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
#  'B' 'LSTAT']

x = dataset.data
y = dataset.target

print(x)
print(x.shape)      #(506, 13)    --> input_dim=13
print(y)
print(y.shape)      #(506,)  벡터

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9,
                                                    random_state=6666)

from sklearn.preprocessing import MinMaxScaler, StandardScaler,MaxAbsScaler, RobustScaler
scaler = MaxAbsScaler()
#scaler = StandardScaler()
#scaler = MaxAbsScaler()
#scaler = RobustScaler()

#MaxAbsScaler 20
#MaxAbsScaler - 19.741
#scaler = MaxAbsScaler() - 22
#StandardScaler()  21, 

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print('x_train :', x_train)
print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))


# print('x_test :', x_test)
# print('y_train :', y_train)
# print('y_test :', y_test)


#2. 모델구성
model = Sequential()
# model.add(Dense(10, input_dim=13))
model.add(Dense(10, input_shape=(13,)))   # input_shape 는 벡터형태로  # 이미지 input_shape=(8,8,1)
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=32,
          validation_split=0.2)
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('로스 : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)
print("걸린시간 : ", round(end - start , 2),"초")

print("============================= hist =========================")
print(hist)
print("============================= hist.history ==================")
print(hist.history)
print("============================= loss ==================")
print(hist.history['loss'])
print("============================= val_loss ==================")
print(hist.history['val_loss'])
print("==========================================================")

import matplotlib.pyplot as plt
plt.rcParams['font.family'] ='Malgun Gothic'
plt.figure(figsize=(9,6))       # 그림판 사이즈
plt.plot(hist.history['loss'], c='red', label='loss')
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.legend(loc='upper right')
plt.title('보스턴 loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()      # 모눈종이
plt.show()


# {'loss': [51104.89453125, 4791.69677734375, 2480.6640625, 1702.1485595703125, 1233.7120361328125, 979.5753173828125, 786.275390625, 628.3425903320312, 506.5628662109375, 412.4776306152344], 'val_loss': [2691.3779296875, 4239.00390625, 1768.7796630859375, 1355.718017578125, 1018.6663208007812, 
# 800.6229858398438, 631.0289306640625, 499.1551513671875, 400.5531921386719, 327.1961975097656]}  =  딕셔너리

# (x, y, train_size=0.8, random_state=3333)  epochs=400, batch_size=5
# 로스 :  19.67884063720703
# r2스코어 :  0.7459871274196113
# (x, y, train_size=0.8, random_state=6666)  epochs=400, batch_size=5
# 로스 :  20.507410049438477
# r2스코어 :  0.7809722401357289
# (x, y, train_size=0.8, random_state=6666)  epochs=500, batch_size=5
# 로스 :  20.05512809753418
# r2스코어 :  0.7858027954982522

print('로스 : ', loss)

# St scaler 로스 :  19.375280380249023
# MM scaler 로스 :  20.29813003540039