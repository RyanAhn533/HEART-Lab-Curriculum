import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

print(x.shape)
print(y.shape)

"""
print(datasets)
print(datasets.DESCR)
print(datasets.feature_names)
print(x.shape, y.shape)
#data를 numpy로 받았으면 describe로 못봄
#남자 여자만 구분하는 것 인줄 알았는데, 알고보니 3가지 분류 필요시?
# 눈으로 확인 해야함 근데 만약 100만개면 어떻게
# numpy에서 y가 0과1 라벨의 종류가 몇가지인지 찾기 pandas에서 
print(np.unique(y,return_counts = True))
print(pd.DataFrame(y).value_counts())
#y로 갯수랑 이름
#0과1의 갯수가 몇개인지 찾아요
print(np.unique(y))
"""

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.7,
                                                    random_state=153)

#2.모델
model = Sequential()
model.add(Dense(32, input_dim=30, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=100,
                   restore_best_weights=True)

model.fit(x_train, y_train, epochs=100, batch_size=8,
          verbose=1, validation_split=0.2,
          callbacks=[es])

#평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_pred = np.round(y_predict)
accuracy_score = accuracy_score(y_test, y_pred)
r2 = r2_score(y_test, y_predict)

print("r2 스코어 :", r2)
print('로스는 ? ', loss)
print("acc_score : ", accuracy_score)
