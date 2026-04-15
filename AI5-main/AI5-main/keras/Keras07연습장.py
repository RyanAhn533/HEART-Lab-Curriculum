import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


x = np.array([range(10), range(21, 31), range(201, 211)])
x=x.T

y = np.array([1,2,3,4,5,6,7,8,9,10])

#2.모델구성

model = Sequential()
model.add(Dense(1, input_dim=3))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=11000, batch_size=200
          )

#결과예측
loss = model.evaluate(x, y)
results = model.predict([[10,31,211]]) 
print('로스 : ', loss)
print('[10,31,211]의 예측값 : ', results)

print(1)
