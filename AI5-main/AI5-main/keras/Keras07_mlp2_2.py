import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1 데이터
x = np.array(range(10))
print(x)
#모든 전산자의 시작은 0이다 / 시작 숫자는 0이다
print(x.shape) #x가 몇개인지 (10,0)

x = np.array(range(1, 11))  #10미만=10직전에 숫자까지=9
print(x)
print(x.shape)

x = np.array([range(10), range(21, 31), range(201, 211)])
print(x)
print(x.shape)
print(x.T)
print(x.T.shape)
#x=x.T
#print(x)
#print(x.shape)

#[실습]
#[10, 31, 211] 예측할것

#1.데이터
#2. 모델구성
#훈련
#예측 결과
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
model.fit(x, y, epochs=100, batch_size=1)

#결과예측
loss = model.evaluate(x, y)
results = model.predict([[10,31,211]]) 
print('로스 : ', loss)
print('[10,31,211]의 예측값 : ', results)
