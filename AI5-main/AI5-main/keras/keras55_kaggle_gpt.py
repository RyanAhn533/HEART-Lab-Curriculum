from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Flatten, Dropout, BatchNormalization
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# 데이터 로드 및 전처리
path = "C:\\프로그램\\ai5\\_data\\kaggle\\jena\\"
a = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)

a = a.head(420407)
b = a.tail(144)
b1 = b.drop(['T (degC)'], axis=1)
x1 = a.drop(['T (degC)'], axis=1)
y1 = a['T (degC)']

# 데이터 스케일링
scaler = MinMaxScaler()
x1 = scaler.fit_transform(x1)
b1 = scaler.transform(b1)

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i: (i + size)]
        aaa.append(subset)
    return np.array(aaa)

size = 48
x = split_x(x1, size)
y = split_x(y1, size)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=3)

# 모델 구성
model = Sequential()
model.add(LSTM(64, input_shape=(size, x.shape[2]), return_sequences=True))
model.add(LSTM(64)) 
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

# 모델 컴파일 및 학습
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=5,
    verbose=1,
    restore_best_weights=True
)

model.fit(x_train, y_train, validation_split=0.2, epochs=100, batch_size=2048, callbacks=[es])

# 평가 및 예측
results = model.evaluate(x_test, y_test)
print('Test Loss : ', results)

x_pred = np.array([b1]).reshape(1, 144, x1.shape[1])
y_pred = model.predict(x_pred)
print(' 예측 결과', y_pred)
print(y_pred.shape)
