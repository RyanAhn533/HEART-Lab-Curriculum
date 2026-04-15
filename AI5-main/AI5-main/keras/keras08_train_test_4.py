import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,7,5,7,8,6,10])

#[실습] 넘파이 리스트의 슬라이싱 !! 7:3

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=123)
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
results = model.predict([[x]])
print("로스 : ", loss)
print('[11]의 예측값 : ', results)

import matplotlib.pyplot as plt
plt.scatter(x, y)
plt.plot(x, results, color='red')
plt.show()