import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터           
 #<- 잘못된 형태        
x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1,1.1,1.2,1.3,1.4,1.5,1.6,1.5,1.4,1.3],
              [9,8,7,6,5,4,3,2,1,0]])  #벡터 2개짜리는 행렬 ->
             #이렇게되면 행열이 반대로 찍힘 
#x = np.array([[1,6],[2,7],[3,8],[4,9],[5,10]])
y = np.array([1,2,3,4,5,6,7,8,9,10])
x = x.T

print(x.shape)
print(y.shape)
#2. 모델구성
# model = Sequential()
model = Sequential()
model.add(Dense(1, input_dim=3))
model.add(Dense(10))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(1))  #y->는 값이 한개니깐

#열, 컴럼, 피처, 특성
#행무시 열우선  10만 by 274 -> imput_dim=274
#5, 100짜리 데이터르 받았어 by -> 100

#3컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
results = model.predict([[10,1.3,0]]) #왜 [[]]이렇게 두개 들어가?
                                # x에 두개들어가서 metrix [5,2]-5는 무시
                                # 근데 predict에 하나만 넣으면 ?
                                # 에러나옴
print('로스 : ', loss)
print('[10, 10]의 예측값 : ', results)

#[실습] : 소수 2째자리까지 맞춰
