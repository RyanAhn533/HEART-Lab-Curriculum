import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])


# [실습] 레이어의 깊이와 노드의 갯수를 이용해서 최소의 loss 맹그러
# 에포는 100으로 고정, 건들지말것!!!
# 로스 기준 0.33 미만!!!!

#2. 모델구성

model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))


x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])   # 한덩어리를    # 두개로 나눠넣으면 훈련이 2번

model.add(Dense(1, input_dim=1)) 이거를
#input_dim=1))   # 여기에 넣음
model.add(Dense(1))   # 모두 하나 하나 하나





model.fit(x, y, epochs=epochs)   # 여기서 훈련하라
model.fit(x, y, epochs=epochs, batch_size=3)   # 3번씩 훈련
model.fit(x, y, epochs=epochs, )    # batch_size=3 
# 사이즈를 정하지 않아도 컴마 찍는것 까지 문제없이 돌아간다.


model.fit(x, y, epochs=epochs, ) # batch_size=3)