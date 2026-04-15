from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time as t
import matplotlib.pyplot as plt
from matplotlib import rc
from tensorflow.keras.callbacks import EarlyStopping

path = 'C:\\프로그램\\ai5\\_data\\bike-sharing-demand\\'
# \a  \b 이런걸 하나의 문자로 인식함 줄바꿈 이런거
# # 위와같은 애들 \ -> \\로 바꿔줘야함 / // 도 가능
# path = 'C:/프로그램//ai5\_data\\bike-sharing-demand'

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sampleSubmission_csv = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

train_csv = train_csv.dropna() #train_csv 데이터에서 결측치 삭제
test_csv = test_csv.fillna(test_csv.mean()) #test_csv에는 결측치 평균으로 넣기

x = train_csv.drop(['count'], axis = 1)
y = train_csv[['count']] #, 'registered


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,
                                                    shuffle=True, random_state=3)

from sklearn.preprocessing import MinMaxScaler, StandardScaler,MaxAbsScaler, RobustScaler
scaler = RobustScaler()
#MaxAbsScaler 로스는 ? 0.30194756388664246
#r2스코어는?  0.9999906821129658
#RobustScaler 로스는 ? 0.26683348417282104
#r2스코어는?  0.9999917657101542

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성
model = Sequential()
model.add(Dense(20, activation='relu', input_dim=10))
model.add(Dense(10, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='linear'))

#컴파일 훈련
model.compile(
    loss='mse',
    optimizer='adam')
es= EarlyStopping(monitor='val_loss', mode = 'min', patience=20,
                  restore_best_weights=True)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
mcp = ModelCheckpoint(
    monitor='val_loss', 
    mode='auto',
    verbose=1,
    save_best_only=True, filepath=("./_save/keras30/keras30_5_save_model.h1"))

model.fit(x_train, y_train, epochs=200, batch_size=128,
          verbose=1, validation_split=0.2, callbacks=[es])
#model.save("./_save/keras30/keras30_5")
model = load_model("./_save/keras30/keras30_5")
#평가예측
loss = model.evaluate(x_test, y_test)
print("로스는 ?", loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2스코어는? ", r2)

# 후 r2 0.9999 로스 0.31
#전 ㄱ2 0.9699 로스 0.11
#Standard  로스는 ? 7.754250526428223
#r2스코어는?  0.9997607093510704
