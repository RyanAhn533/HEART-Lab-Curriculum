# 28_2 copy

import sklearn as sk
from sklearn.datasets import load_boston
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time

#1. 데이터 
dataset = load_boston()
print(dataset)
print(dataset.DESCR) 
print(dataset.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
#  'B' 'LSTAT']

x = dataset.data    # x데이터 분리
y = dataset.target  # y데이터 분리, sklearn 문법

print(x)
print(x.shape)  # (506, 13)
print(y)
print(y.shape)  # (506, )

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=231)


#2. 모델 구성
model = Sequential()
# model.add(Dense(32, input_dim=13))
model.add(Dense(10, input_shape=(13,)))      # 이미지 : input_shape=(8,8,1)
model.add(Dense(5))
model.add(Dense(1))

model.summary()

model.save("./_save/keras28/keras28_1_save_model.h5")




#3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')
# start = time.time()
# hist = model.fit(x_train, y_train, epochs=10, batch_size=16,
#           verbose=3, 
#           validation_split=0.1
#           )
# end = time.time()
"""
#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)    # 추가
print('loss :', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 score :', r2)
"""