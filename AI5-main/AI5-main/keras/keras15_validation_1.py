#08-1에서 때옴 

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
#x = np.array([1,2,3,4,5,6,7,8,9,10])
#y = np.array([1,2,3,4,5,6,7,8,9,10]) 

x_train = np.array([1,2,3,4,5,6])
y_train = np.array([1,2,3,4,5,6])

x_val = np.array([7, 8])
y_val = np.array([7, 8])

x_test = np.array([8,9,10])
y_test = np.array([8,9,10])

#이친구들은 평가만함
#Validation은 훈련에 들어감 -> 성능에 형상을 미친다? O X 중 O이지만 크게 미치지 않는다.
#x
#2. 모델구성
model = Sequential()

model.add(Dense(1, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=1, 
          verbose=1, 
          validation_data=(x_val, y_val), ) # 이 파일에서 요놈만 추가

#4. 평가, 예측
print("+++++++++++++++++++++++++++++++++++++++++")
loss = model.evaluate(x_test, y_test)
results = model.predict([11])
print("로스 : ", loss)
print('[11]의 예측값 : ', results)
