from sklearn.datasets import load_boston   
import numpy as np
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 데이터 로드 및 전처리
dataset = load_boston()
x = dataset.data    
y = dataset.target  

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=231)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(type(x))
print(type(y))

# 모델 구성 (함수형)
input1 = Input(shape=(13,))
dense1 = Dense(32, activation='relu', name='ys1')(input1)
dropout1 = Dropout(0.2)(dense1)
dense2 = Dense(16, activation='relu', name='ys2')(dropout1)
dropout2 = Dropout(0.2)(dense2)
dense3 = Dense(8, activation='relu', name='ys3')(dropout2)
dense4 = Dense(4, activation='relu', name='ys4')(dense3)
output1 = Dense(1, name='ys5')(dense4)

model = Model(inputs=input1, outputs=output1)

# 컴파일 및 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, restore_best_weights=True)

# 모델 체크포인트 파일 경로 설정
filepath = './save/mcp2/keras32/keras32-3_mcp3.hdf5'
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

# 모델 훈련
start = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=16, verbose=1, validation_split=0.1, callbacks=[es, mcp])
end = time.time()

# 모델 평가 및 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 score :', r2)


from sklearn.datasets import load_boston   
import numpy as np
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
from tensorflow.keras.models import Sequential, load_model
#1. 데이터 
dataset = load_boston()

x = dataset.data    
y = dataset.target  


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=231)

from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

"""
1
#2-1. 모델 구성(순차형)
model = Sequential()
model.add(Dense(32, input_dim=13, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))
"""

#2-2 모델구성(함수형)
input1 = Input(shape=(13,))
dense1 = Dense(32)(input1, name='ys1')(input1)
dense2 = Dense(16)(dense1, name='ys2')(dense1)
dropout1 = Dropout(0.3)(dense1)
dense3 = Dense(8)(dense2, name='ys3')(dropout1)
dense4 = Dense(4)(dense3, name='ys4')(dense3)
output1 = Dense(1)(dense4, name='ys5')(dense4)
#시작 input1 ~ output1 여기까지가 모델이야라고 정해주는게 필요함 아래 모델
model = Model(inputs=input1, outputs=output1)


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=10, restore_best_weights=True)

filepath = './save/mcp2/keras32/keras32-3_mcp3.hdf5'
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)


start = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=16,
          verbose=1, 
          validation_split=0.1, callbacks=[es, mcp]
          )

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)    # 추가
print('loss :', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 score :', r2)

#파일명이 너무 길어지면 가독성이 좋지 않기 때문에, 본인이 가독성이 좋게끔 조절