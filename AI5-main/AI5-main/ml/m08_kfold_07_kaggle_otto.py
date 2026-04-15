#url : https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data?select=test.csv
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats
import scipy as sp
import scipy.stats
from sklearn.model_selection import KFold, StratifiedKFold

path = "C:/프로그램/ai5/_data/kaggle/otto/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
samplesubmission1_csv = pd.read_csv(path + "samplesubmission.csv", index_col=0)

print(train_csv.select_dtypes(include=['object']).columns)
print(test_csv.select_dtypes(include=['object']).columns)

train_csv.info()
test_csv.info()
print(train_csv['target'].value_counts())
train_csv['target'] = train_csv['target'].replace({'Class_1' : 1, 'Class_1' : 1, 'Class_2' : 2, 'Class_3' : 3, 'Class_4' : 4, 'Class_5' : 5, 'Class_6' : 6, 'Class_7' : 7, 'Class_8' : 8, 'Class_9' : 9, })



x = train_csv.drop(['target'], axis=1)

scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

y = train_csv['target']


x = np.array(x)
y = np.array(y)

# KFold 설정
n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=333)

# 2. 모델 훈련 및 평가
r2_scores = []
losses = []
accuracys = []
for train_index, test_index in kfold.split(x, y):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]



#모델
    model = Sequential()
    model.add(Dense(512, input_dim=93, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))

#컴파일 훈련
    model.compile(loss='mse',
    optimizer='adam',
    metrics=['acc'])
    es= EarlyStopping(monitor='val_loss', mode = 'min', patience=2,
                  restore_best_weights=True)

    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    mcp = ModelCheckpoint(monitor='val_loss', 
    mode='auto',
    verbose=0,
    save_best_only=True, filepath=("./_save/ml08/keras32_13_save_model.h1"))

    model.fit(x_train, y_train, epochs=100, batch_size=1024,
          verbose=0, validation_split=0.2, callbacks=[es])



#평가예측


    loss = model.evaluate(x_test, y_test)
    losses.append(loss)

    y_pred = model.predict(x_test)
    y_pred = np.round(y_pred)
    r2 = r2_score(y_test, y_pred)
    acc_score = accuracy_score(y_test, y_pred)
    accuracys.append(acc_score)

    print('로스값은 : ', loss)
    print('정확도는 : ', accuracy_score)
    print("R2 스코어 :", r2)

# KFold의 평균 결과 출력
print(f"\n최종 평균 로스: {np.mean(losses)}")
print(f"최종 평균 r2스코어: {np.mean(r2_scores)}")
print(f"최종 평균 r2스코어: {np.mean(accuracys)}")

'''
로스값은 :  [7.899733543395996, 0.03118939884006977]
정확도는 :  <function accuracy_score at 0x00000166795EF670>
R2 스코어 : -0.2637758255004883
387/387 [==============================] - 1s 1ms/step - loss: 8.5010 - acc: 0.0312
로스값은 :  [8.50096607208252, 0.03118939884006977]
정확도는 :  <function accuracy_score at 0x00000166795EF670>
R2 스코어 : -0.36363375186920166
387/387 [==============================] - 1s 1ms/step - loss: 8.3431 - acc: 0.0312
로스값은 :  [8.34311580657959, 0.03118939884006977]
정확도는 :  <function accuracy_score at 0x00000166795EF670>
R2 스코어 : -0.33901143074035645
387/387 [==============================] - 1s 1ms/step - loss: 6.6352 - acc: 0.0312
로스값은 :  [6.635232448577881, 0.03119191899895668]
정확도는 :  <function accuracy_score at 0x00000166795EF670>
R2 스코어 : -0.07989120483398438
387/387 [==============================] - 1s 1ms/step - loss: 8.6339 - acc: 0.0311
로스값은 :  [8.633877754211426, 0.031111111864447594]
정확도는 :  <function accuracy_score at 0x00000166795EF670>
R2 스코어 : -0.38116371631622314

최종 평균 로스: 4.016879685223103
최종 평균 r2스코어: nan
최종 평균 r2스코어: 0.21333974913974912
'''