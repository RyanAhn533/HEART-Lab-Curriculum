from sklearn.datasets import load_boston   
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
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

print(x_train)
print(np.min(x_train), np.max(x_train))    
print(x_test)
print(np.min(x_test), np.max(x_test))     


#2. 모델 구성
model = Sequential()
model.add(Dense(32, input_dim=13, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=10, restore_best_weights=True)

####################### mcp 세이브 파일명 만들기 시작#########################
import datetime
data = datetime.datetime.now()
print(data)
print(type(data))
data = data.strftime("%m%d_%H%M")
print(type(data))
print(data)


path = './save/mcp2/keras32/'

filename='{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = "".join([path, 'k32_',data,'_', filename])

#생성 예 : ./_save/keras29_mcp/k29_1000~0.7777.dfg5

#파일 이름이 최고의 에포일 때로 저장되게끔!
#파일 이름보고 성능이 안좋으면 지워버리기
#파이썬에서 붙히는 거 문법"".join()
####################### mcp 세이브 파일명 만들기 끗#########################

mcp = ModelCheckpoint(
    monitor='val_loss', 
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath='./_save/keras32/keras32-3_mcp3.hdf5'
)
start = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=16,
          verbose=1, 
          validation_split=0.1, callbacks=[es, mcp]
          )
end = time.time()
#model.save('./_save/keras29_mcp/keras29_3_save_model.h5')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)    # 추가
print('loss :', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 score :', r2)

#파일명이 너무 길어지면 가독성이 좋지 않기 때문에, 본인이 가독성이 좋게끔 조절