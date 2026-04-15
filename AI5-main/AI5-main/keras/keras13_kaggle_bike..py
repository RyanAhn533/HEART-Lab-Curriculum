#url1 = 'https://www.kaggle.com/competitions/bike-sharing-demand/data?select=test.csv'
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

path = 'C:\\프로그램\\ai5\\_data\\bike-sharing-demand\\'
# \a  \b 이런걸 하나의 문자로 인식함 줄바꿈 이런거
# # 위와같은 애들 \ -> \\로 바꿔줘야함 / // 도 가능
# path = 'C:/프로그램//ai5\_data\\bike-sharing-demand'

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sampleSubmission_csv = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

print(train_csv.shape)
print(test_csv.shape)
print(sampleSubmission_csv.shape)
#train.csv -> date time -> 복잡하니 index로 인식시키겠다.

print(train_csv.columns)
#Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
    #    'humidity', 'windspeed', 'casual', 'registered', 'count'],
    #   dtype='object')
print(train_csv.info())
print(test_csv.info())

print(train_csv.describe())
##############결측치 확인 ###############
print(train_csv.isna().sum())
print(test_csv.isnull().sum())

######x와y를 분리
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=433, train_size=0.8)


#모델구성
model = Sequential()
model.add(Dense(8,activation='relu', input_dim=8))    
model.add(Dense(64,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1, activation='linear'))
 #난 한정시킬거야 다음레이어에 전달하는 값을
#w값때매 자꾸 음수나오면 activation function을 넣자 relu 함수를 넣자
#relu함수를 넣으면 데이터가 0이상의 값이나오면 그냥하세요 0이하면 하지마세요
#그러면 다음 레이어에 전달하는 값은 전부 양수가 된다.

#컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=64)

#결과
loss = model.evaluate(x_test, y_test)
print('로스 : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)

y_submit = model.predict(test_csv)
sampleSubmission_csv['count'] = y_submit
sampleSubmission_csv.to_csv(path + "bike_submission_0717_relu real fin_5.csv")