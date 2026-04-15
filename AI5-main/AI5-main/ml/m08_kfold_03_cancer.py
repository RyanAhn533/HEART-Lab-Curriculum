import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time as t
import matplotlib.pyplot as plt
from matplotlib import rc
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

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
    model.add(Dense(128, input_dim=30, activation='relu'))
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
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
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
4/4 [==============================] - 0s 667us/step - loss: 0.2303 - acc: 0.9649
로스는 ? [0.23028913140296936, 0.9649122953414917]
r2스코어는?  0.8562337642919293
4/4 [==============================] - 0s 333us/step - loss: 0.0929 - acc: 0.9649
로스는 ? [0.09286573529243469, 0.9649122953414917]
r2스코어는?  0.8860307539538271
4/4 [==============================] - 0s 667us/step - loss: 0.0891 - acc: 0.9737
로스는 ? [0.08911409229040146, 0.9736841917037964]
r2스코어는?  0.8945718404126071
4/4 [==============================] - 0s 333us/step - loss: 0.1215 - acc: 0.9737
로스는 ? [0.12151947617530823, 0.9736841917037964]
r2스코어는?  0.8825436786730869
4/4 [==============================] - 0s 333us/step - loss: 0.0369 - acc: 0.9823
로스는 ? [0.03685814142227173, 0.982300877571106]
r2스코어는?  0.9518064928632953

최종 평균 로스: 0.5430140428245067
최종 평균 r2스코어: 0.8942373060389492
'''