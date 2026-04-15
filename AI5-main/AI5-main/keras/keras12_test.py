#https://dacon.io/competitions/open/235576/overview/description

import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


path = "./_data/따릉이/"
train_csv = pd.read_csv(path + "train.csv", index_col=0) 

test_csv = pd.read_csv(path + "test.csv", index_col=0) 

submission_csv = pd.read_csv(path + "submission.csv", index_col=0)

train_csv = train_csv.dropna()

test_csv = test_csv.fillna(test_csv.mean())

x = train_csv.drop(['count'], axis=1)

y = train_csv['count']


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.95, shuffle=True, random_state=4343
                                                    )

#모델구성
model = Sequential()
model.add(Dense(1,activation='relu', input_dim=9))
model.add(Dense(16,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(1))

#컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=32)

#평가예측
loss = model.evaluate(x_test, y_test)
print("로스 : ", loss)

y_predict = model.predict(x_test)  #예측하기 위한 가중치 들어감, 최적의 가중치 - 최적의 loss에 최적의 w(가중치)
r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)

y_submit = model.predict(test_csv)
print(y_submit)
print(y_submit.shape)   #(715, 1)
######################   submission.csv 만들기 // count 컬럼에 값 넣어주면 된다 ####
submission_csv['count'] = y_submit
print(submission_csv)
print(submission_csv.shape)

submission_csv.to_csv(path + "submission_relu28.csv") 

#2683.39306640625
#r2스코어 :  0.6216009595565672 

#로스 :  2646.247802734375 / 1 8 10 8 1   3165