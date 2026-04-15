#https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016

#y 는 T(degC) 로 잡아라.
#자르는 거 맘대로

#predict해야할 부분
#31.12.2016 00:10:00 ~ 01.01.2017 00:00:00 까지 맞춰라 y 144개
#None , 144
#y의 shape는 (n,144)

#url : https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data?select=test.csv
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Input, BatchNormalization
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
import time 
from sklearn.model_selection import train_test_split

path = "C:\\프로그램\\ai5\\_data\\kaggle\\jena\\"

a = pd.read_csv(path + 'jena_climate_2009_2016_1.csv', index_col=0)
print(a.shape)


x=a.drop(['T (degC)'], axis=1)

y=a['T (degC)']

a1 = x[:420408]
a2 = x[420408:420551]
print(a1.shape)
print(a2.shape)
a3 = y[:420408]
a4 = y[420408:420551]


from sklearn.preprocessing import MinMaxScaler, StandardScaler,MaxAbsScaler, RobustScaler

scaler = StandardScaler()
print(x)
print(y)
print(x.shape)
print(y.shape)


#2. 모델구성
model = Sequential()

model.add(LSTM(64, input_shape=(12,1), return_sequences=True)) #3은 time steps, 1은 features
model.add(LSTM(64,)) 
model.add(Dense(32,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Flatten()) 
model.add(Dense(8,activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(
    monitor = 'loss',
    mode = 'min',
    verbose=1,
    patience=1000,
    restore_best_weights=True
)

model.fit(x,y, epochs=10, batch_size=2048, callbacks=[es])
#4. 평가, 예측
results = model.evaluate(x,y)
print('loss : ', results)

x_pred = np.array([]).reshape(1,5,2) #[[[8]]]
#벡터형태 데이터 (3,) -> (1,3,1)
#스칼라는 행렬아님
y_pred = model.predict(x_pred)
print(' 결과', y_pred)
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 데이터 로드
path = "C:\\프로그램\\ai5\\_data\\kaggle\\jena\\/jena_climate_2009_2016.csv"  # 경로를 적절히 수정하세요.
data = pd.read_csv(path, index_col=0)

# 'T (degC)' 열이 목표 변수 (y), 나머지 열들은 입력 변수 (X)
y = data['T (degC)']
X = data.drop(columns=['T (degC)'])

# 데이터 스케일링
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 시계열 데이터 형태로 변환
def create_sequences(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 48  # 예시: 48시간(2일) 단위로 예측
X_seq, y_seq = create_sequences(X_scaled, y.values, time_steps)

# 훈련 및 테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# LSTM 모델 구성
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

# 모델 컴파일
model.compile(optimizer='adam', loss='mse')

# 모델 학습
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, 
                    validation_split=0.1, 
                    epochs=50, 
                    batch_size=64, 
                    callbacks=[early_stop],
                    verbose=1)

# 모델 평가
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# 미래 데이터 예측
# 예측할 데이터 X_future를 준비
# 예시로 X_test의 마지막 시퀀스를 기반으로 다음 값을 예측합니다.
X_future = X_test[-1].reshape(1, X_test.shape[1], X_test.shape[2])
y_pred = model.predict(X_future)
print(f'예측된 T (degC): {y_pred[0]}')
# 예측된 결과를 저장합니다.
submission = pd.DataFrame({
    'id': range(len(y_pred)),
    'T (degC)': y_pred.flatten()
})

submission.to_csv('jena_climate_predictions.csv', index=False)
