#01부터 13까지 쭉 카피해서
#gpu 이래와 cpu 일때의 시간을 체크해서 비교할 것
from sklearn.datasets import load_boston   
import numpy as np
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import tensorflow as tf
print(tf.__version__)
#physical_devisces 물리적인 경로
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)

# 데이터 로드 및 전처리
dataset = load_boston()
x = dataset.data    
y = dataset.target  

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=231)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

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
print("걸린시간은?", "gpu on" if (len(gpus) > 0) else "gpu off", round(end - start, 2), '초')
if(gpus):
    print("쥐피유 돈다!!!")
else:
    print("쥐피유 없다!")
#걸린시간은? gpu on 4.74 초
#걸린시간은? gpu off 1.62 초
"""
걸린시간은? gpu on 4.9 초
쥐피유 돈다!!!

걸린시간은? gpu off 34.33 초
쥐피유 없다!
"""