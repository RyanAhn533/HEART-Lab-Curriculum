from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


path = "./_data/따릉이/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "submission.csv", index_col=0)
#path = "./_data/따릉/" 이렇게 이용해서 pd구문 안을 짧게 만들 수 있음

train_csv = train_csv.dropna() #train_csv 데이터에서 결측치 삭제
test_csv = test_csv.fillna(test_csv.mean()) #test_csv에는 결측치 평균으로 넣기
x = train_csv.drop(['count'], axis=1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=5, train_size=0.7)
x_train, x_val, y_train, y_val = train_test_split(x, y, shuffle=True, random_state=5, train_size=0.2)
#모델구성

model = Sequential()
model.add(Dense(1, input_dim=9))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#컴파일 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size = 10, verbose=1, validation_split=0.3)

#평가예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

#내보내기
y_submit = model.predict(test_csv)
submission_csv['count'] = y_submit
submission_csv.to_csv(path + "submission_0718.csv")