from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, LSTM
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import numpy as np
import time as t
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import matplotlib.pyplot as plt

##############
# 데이터 로드 및 전처리
dataset = load_boston()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['target'] = dataset['target']
#df.boxplot()  #CRIM ZN B
#df.plot.box()

x = df.drop(['target'], axis=1).copy()
y = df['target']
print(x.shape, y.shape)

x['CRIM'] = np.log1p(x['CRIM'])
x['ZN'] = np.log1p(x['ZN'])
x['B'] = np.log1p(x['B'])


# 훈련 및 테스트 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=3)
print(x_train.shape)
print(x_test.shape)
y_train = np.log1p(y_train)
y_test = np.log1p(y_test)

model = Sequential()
model.add(LSTM(10, input_shape=(13, 1))) # timesteps , features
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(2048, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(1))
model.summary()

# 모델 컴파일 및 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=1000, batch_size=32, verbose=2, validation_split=0.3, callbacks=[es])

# 평가 및 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

# R2 스코어 계산
r2 = r2_score(y_test, y_predict)
print("로스 :", loss)
print("R2 스코어:", r2)

#R2 스코어: 0.7796704150990024