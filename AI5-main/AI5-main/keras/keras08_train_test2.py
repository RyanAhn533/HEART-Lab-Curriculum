import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

#[실습] 넘파이 리스트의 슬라이싱 !! 7:3

x_train = x[0:7]   # = x_train = x[:-3] = x_train = x[0:-3]
y_train = y[0:7]
print(x_train)
print(y_train)
x_test = x[7:11]   # = x_test = x[7:]  = x_test = x[-3:]
y_test = y[7:11]
print(x_test)
print(y_test)

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4.평가, 예측
print("++++++++++++++++++++")
loss = model.evaluate(x_test, y_test)
results = model.predict([11])
print("로스 : ", loss)
print('[11]의 예측값 : ', results)