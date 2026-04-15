from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, LSTM
import sklearn as sk
from sklearn.datasets import load_wine
import numpy as np
import time as t
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
##############
import warnings
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
###############
#1.데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

scaler = StandardScaler()
x = scaler.fit_transform(x)

print(x.shape)
print(y.shape)

# x = x.reshape(178,13,1,1)
x = np.array(x)
y = np.array(y)
print(x.shape, y.shape)

# 스케일링 적용
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

# KFold 설정
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)

# 2. 모델 훈련 및 평가
r2_scores = []
losses = []

for train_index, test_index in kfold.split(x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model = Sequential()
    model.add(Conv1D(filters=10, kernel_size=2, input_shape=(13, 1)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1))

    model.compile(
        loss='mse',
        optimizer='adam',
        metrics=['acc'])

    es = EarlyStopping(monitor='val_loss', mode='min', patience=20, 
                       restore_best_weights=True)
    
    mcp = ModelCheckpoint(
        monitor='val_loss',
        mode='auto',
        verbose=1,
        save_best_only=True, 
        filepath=("./_save/keras32/keras32_dropout.h1"))

    model.fit(x_train, y_train, epochs=200, batch_size=128, 
              verbose=0, validation_split=0.2, callbacks=[es])

    # 평가 및 예측
    loss = model.evaluate(x_test, y_test)
    losses.append(loss)
    y_predict = model.predict(x_test)
    r2 = r2_score(y_test, y_predict)
    r2_scores.append(r2)
    
    print("로스는 ?", loss)
    print("r2스코어는? ", r2)

# KFold의 평균 결과 출력
print(f"\n최종 평균 로스: {np.mean(losses)}")
print(f"최종 평균 r2스코어: {np.mean(r2_scores)}")
'''
로스는 ? [0.07790739834308624, 0.6666666865348816]
r2스코어는?  0.8662675619125366
2/2 [==============================] - 0s 2ms/step - loss: 0.2540 - acc: 0.5833
로스는 ? [0.254010409116745, 0.5833333134651184]
r2스코어는?  0.5822365880012512
2/2 [==============================] - 0s 2ms/step - loss: 0.1276 - acc: 0.6389
로스는 ? [0.12758393585681915, 0.6388888955116272]
r2스코어는?  0.8086240887641907
2/2 [==============================] - 0s 35ms/step - loss: 0.1706 - acc: 0.7429
로스는 ? [0.1705714464187622, 0.7428571581840515]
r2스코어는?  0.6945174932479858
2/2 [==============================] - 0s 2ms/step - loss: 0.0626 - acc: 0.7143
WARNING:tensorflow:5 out of the last 9 calls to <function Model.make_predict_function.<locals>.predict_function at 0x0000023642D28DC0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
로스는 ? [0.06258315593004227, 0.7142857313156128]
r2스코어는?  0.8879175782203674

최종 평균 로스: 0.40386881306767464
최종 평균 r2스코어: 0.7679126620292663
'''