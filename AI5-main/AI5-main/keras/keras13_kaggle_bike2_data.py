# 기존 캐글 데이터에서

# 1. train)csv의 y를 casual과 register로 잡는다.
# 이후 훈련을 해서 test_csv의 casual과 register를 predict한다
# 2. test_csv에 casual과 register 컬럼을 합쳐서
# 3. train_csv에 y를 count로 잡는다.
# 4. 전체 훈련
# 5. test_csv 예측해서 submission에 붙여!!

#환경구축
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# 1. 데이터
path = 'C:\\프로그램\\ai5\\_data\\bike-sharing-demand\\'
# \a  \b 이런걸 하나의 문자로 인식함 줄바꿈 이런거
# # 위와같은 애들 \ -> \\로 바꿔줘야함 / // 도 가능
# path = 'C:/프로그램//ai5\_data\\bike-sharing-demand'

train_csv = pd.read_csv(path + "train.csv", index_col=0) #첫번째 열은 시간이니 인덱스로
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sampleSubmission_csv = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

x = train_csv.drop(['casual', 'registered', 'count'], axis = 1)
y = train_csv[['casual', 'registered']] #두가지의 list이기에 대괄호가 2개

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=433, train_size=0.7)

print('x_train : ', x_train.shape)
print('x_test : ', x_test.shape)
print('y_train : ', y_train.shape)
print('y_test : ', y_test.shape)

"""
x_train :  (7620, 8)
x_test :  (3266, 8)
y_train :  (7620, 2)
y_test :  (3266, 2) """


#모델구성
model = Sequential()
model.add(Dense(4,activation='relu', input_dim=8))    
model.add(Dense(8,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(2, activation='linear'))


#컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=20, batch_size=10)

#결과
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
#print(y_predict)
y_submit = model.predict(test_csv)
"""
print(test_csv.shape)   #(64963, 8)
print(y_submit)
print(y_submit.shape)   #(6493, 2)
print("test_csv의 타입 은? : ",type(test_csv)) #pandas.core.frame.dataFrame
print("y_submit의 타입 은? : ",type(y_submit)) #numpy.ndarray

print(y_submit.shape)

test2_csv = test_csv
print(test2_csv.shape) #(6493, 8)

test2_csv[['casual', 'resistered']] = y_submit
print(test2_csv)

test2_csv.to_csv(path + "bike_sub_resister sep1.csv") 
"""




casual_predict = y_submit[:,0]
registered_predict = y_submit[:,1]
test_csv = test_csv.assign(casual=casual_predict, registered = registered_predict)
test_csv.to_csv(path + "test_columnplus.csv")

print(y_submit.shape)

test2_csv = test_csv
print(test2_csv.shape) #(6493, 8)

test2_csv[['casual', 'resistered']] = y_submit
print(test2_csv) 