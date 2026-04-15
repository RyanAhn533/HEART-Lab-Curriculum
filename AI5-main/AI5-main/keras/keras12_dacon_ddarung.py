#https://dacon.io/competitions/open/235576/overview/description

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#인덱스와 칼럼명은 실질적인 데이터가 아니다

#1. 데이터
#괄호가 있으면 함수, /read처럼 소문자면 함수, 대문자면 클래스라는 암묵적인 룰
path = "./_data/따릉이/"
train_csv = pd.read_csv(path + "train.csv", index_col=0) 
#print(train_csv)  #1459rows x 10 columns

#index_col=0 의 의미 0번째에 있는 열을 인덱스라고 칭한다

test_csv = pd.read_csv(path + "test.csv", index_col=0) 
#print(test_csv) #715 rows x 9columns

submission_csv = pd.read_csv(path + "submission.csv", index_col=0)
#print(submission_csv)

#항상 틀리는 거 1. 오타 2. 경로 3. shape
#"1"이렇게 써있으면 문자다. "1" + "A" = "1A"
#sklearn 에서는 data.target, data.data 이렇게 분리했는데
#그건 잊어버려라! Pandas가 굉장히 중요!

"""
print(train_csv.shape) # 1459 rows x 10columns
print(test_csv.shape) # (715, 9)
print(submission_csv.shape)
print(train_csv.columns) """

#Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#       'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',        
#      'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#     dtype='object')

#print(train_csv.info())
#x는 결측치가 있어도 되지만 y는 결측치가 있으면 안된다.
#  0   hour                    1459 non-null   int64
#  1   hour_bef_temperature    1457 non-null   float64
#  2   hour_bef_precipitation  1457 non-null   float64
#  3   hour_bef_windspeed      1450 non-null   float64
#  4   hour_bef_humidity       1457 non-null   float64
#  5   hour_bef_visibility     1457 non-null   float64
#  6   hour_bef_ozone          1383 non-null   float64
#  7   hour_bef_pm10           1369 non-null   float64
#  8   hour_bef_pm2.5          1342 non-null   float64
#  9   count                   1459 non-null   float64

#############결측치 처리 1. 삭제###################
#print(train_csv.isnull().sum())
#print(train_csv.isna().sum())

train_csv = train_csv.dropna()   #train_csv.dropna() -> na를 떨군다
#print(train_csv.isna().sum())
#print(train_csv)

#print(test_csv.info())
""" 0   hour                    715 non-null    int64
 1   hour_bef_temperature    714 non-null    float64
 2   hour_bef_precipitation  714 non-null    float64
 3   hour_bef_windspeed      714 non-null    float64
 4   hour_bef_humidity       714 non-null    float64
 5   hour_bef_visibility     714 non-null    float64
 6   hour_bef_ozone          680 non-null    float64
 7   hour_bef_pm10           678 non-null    float64
 8   hour_bef_pm2.5          679 non-null    float64"""
 #test_csv에서는 결측치를 삭제할 수 없는 이유 = submission
 # 결과값을 예측할 때 순서가 맞지 않는다. 
# 결측치에는 평균값을 넣는다!

test_csv = test_csv.fillna(test_csv.mean())
#print(test_csv.info())

x = train_csv.drop(['count'], axis=1) 
#axis=0은 행방향(가로)로 이동, axis=1은 열방향으로 이동
#train_csv에서 count라는 열을 지운걸 x라고 정의한다.
#print(x) #1328 rows x 9 columns
y = train_csv['count']
#print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=123)

#모델구성
model = Sequential()
model.add(Dense(1, input_dim=9))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=10)

#평가예측
loss = model.evaluate(x_test, y_test)
#print("로스 : ", loss)
"""
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("r2스코어 :", r2)"""

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)

"""
y_submit = model.predict(test_csv)
print(y_submit)
print(y_submit.shape)   #(715, 1)
######################   submission.csv 만들기 // count 컬럼에 값 넣어주면 된다 ####
submission_csv['count'] = y_submit
print(submission_csv)
print(submission_csv.shape)

submission_csv.to_csv(path + "submission_0716_1728.csv") """