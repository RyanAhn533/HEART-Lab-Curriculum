#08-1
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
# x = np.array([1,2,3,4,5,6,7,8,9,10])
# y = np.array([1,2,3,4,5,6,7,8,9,10]) 

x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,5,6,7])

x_test = np.array([8,9,10])
y_test = np.array([8,9,10])


#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#epoch 과정이 아래에 나열되는 것은 사람에게 보여주기 위해서
# 시간 낭비가 있으니 나오지 않게 하는 것이 더 효율적
#그렇게 설정하는 법은?

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=2) # 0넣으면 안보이고 1넣으면 기본값이라서 터미널에 나옴
"""
verbos=0 : 침묵
verbos=1 : 디폴트
verbos=2 : 프로그래스바
verbos=나머지 : 에포만 나온다


2 넣으면 진행바 안나옴 Epoch 99/100
7/7 - 0s - loss: 0.0044 - 3ms/epoch - 427us/step
Epoch 100/100
7/7 - 0s - loss: 0.0042 - 2ms/epoch - 285us/step
+++++++++++++++++++++++++++++++++++++++++
1/1 [==============================] - 0s 55ms/step - loss: 0.0157
로스 :  0.015699824318289757
[11]의 예측값 :  [[10.818541]]
PS C:\프로그램\ai5> 

3이상은 epochs만나옴

"""

#4. 평가, 예측
print("+++++++++++++++++++++++++++++++++++++++++")
loss = model.evaluate(x_test, y_test)
results = model.predict([11])
print("로스 : ", loss)
print('[11]의 예측값 : ', results)


# 7/7 [==============================] - 0s 332us/step - loss: 0.0153    # 훈련
# Epoch 100/100
# 7/7 [==============================] - 0s 332us/step - loss: 0.0144
# +++++++++++++++++++++++++++++++++++++++++
# 1/1 [==============================] - 0s 58ms/step - loss: 0.0592    # 평가
# 로스 :  0.059226710349321365
# [11]의 예측값 :  [[10.650603]]