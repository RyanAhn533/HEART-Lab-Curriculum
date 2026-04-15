# 기존 캐글 데이터에서

# 1. train)csv의 y를 casual과 register로 잡는다.
# 이후 훈련을 해서 test_csv의 casual과 register를 predict한다
# 2. test_csv에 casual과 register 컬럼을 합쳐서
# 3. train_csv에 y를 count로 잡는다.
# 4. 전체 훈련
# 5. test_csv 예측해서 submission에 붙여!!

#환경구축
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 1. 데이터
path = 'C:\\프로그램\\ai5\\_data\\bike-sharing-demand\\'
# \a  \b 이런걸 하나의 문자로 인식함 줄바꿈 이런거
# # 위와같은 애들 \ -> \\로 바꿔줘야함 / // 도 가능
# path = 'C:/프로그램//ai5\_data\\bike-sharing-demand'

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test2_csv = pd.read_csv(path + "test2.csv", index_col=0)
sampleSubmission_csv = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

print(train_csv.shape)
print(test2_csv.shape)
print(sampleSubmission_csv.shape)

x = train_csv.drop(['count'], axis = 1)
y = train_csv['count'] #, 'registered'

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=433, train_size=0.9)

#모델구성
model = Sequential()
model.add(Dense(16,activation='relu', input_dim=10))
model.add(Dense(32,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1, activation='linear'))


#컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=762, verbose=0)

#결과
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)
#print(y_predict)
y_submit = model.predict(test2_csv)
sampleSubmission_csv['count'] = y_submit
sampleSubmission_csv.to_csv(path + "submission_0718_1423.csv")
