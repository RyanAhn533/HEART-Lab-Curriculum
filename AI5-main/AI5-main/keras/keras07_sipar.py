import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([[1,2,3,4,5,6,7,8,9,10], 
              [10,9,8,7,6,5,4,3,2,1], 
              [9,8,7,6,5,4,3,2,1,0]])
x = x.T
y = np.transpose(y)

#2. 모델
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(3))

#3. 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

#결과예측
loss = model.evaluate(x, y)
results = model.predict([10])
print('로스 : ', loss)
print('[10]의 예측값', results)