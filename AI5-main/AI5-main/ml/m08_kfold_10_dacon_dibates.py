from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, LSTM
import sklearn as sk
import numpy as np
import time as t
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input
##############
import warnings
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
###############

path = "./_data/dacon/diabets/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sample_submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)
#path = "./_data/따릉/" 이렇게 이용해서 pd구문 안을 짧게 만들 수 있음

print(train_csv.columns)
print(test_csv.columns)


x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']

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
    model.add(Conv1D(filters=10, kernel_size=2, input_shape=(8, 1)))
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
로스는 ? [0.2028312236070633, 0.7099236845970154]
r2스코어는?  0.16287004947662354
5/5 [==============================] - 0s 1ms/step - loss: 0.1672 - acc: 0.7252
로스는 ? [0.16718871891498566, 0.7251908183097839]
r2스코어는?  0.26620805263519287
5/5 [==============================] - 0s 6ms/step - loss: 0.1563 - acc: 0.7769
로스는 ? [0.15634329617023468, 0.7769230604171753]
r2스코어는?  0.3092283606529236
5/5 [==============================] - 0s 2ms/step - loss: 0.1674 - acc: 0.7769
로스는 ? [0.16737833619117737, 0.7769230604171753]
r2스코어는?  0.23466086387634277
5/5 [==============================] - 0s 2ms/step - loss: 0.1494 - acc: 0.7692
로스는 ? [0.149435892701149, 0.7692307829856873]
r2스코어는?  0.3079017996788025

최종 평균 로스: 0.4601368874311447
최종 평균 r2스코어: 0.25617382526397703
'''