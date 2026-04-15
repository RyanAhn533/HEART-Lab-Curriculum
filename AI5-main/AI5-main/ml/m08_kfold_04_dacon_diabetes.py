from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from matplotlib import rc
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time as t
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import KFold

path = "./_data/dacon/diabets/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sample_submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)
#path = "./_data/따릉/" 이렇게 이용해서 pd구문 안을 짧게 만들 수 있음

print(train_csv.columns)
print(test_csv.columns)


x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']

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
    model.add(Dense(128, input_dim=8, activation='relu'))
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
        filepath=("./_save/ml08/ml08_diabetes.h5"))

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
5/5 [==============================] - 0s 500us/step - loss: 0.2107 - acc: 0.6641
로스는 ? [0.2106669545173645, 0.6641221642494202]
r2스코어는?  0.13053025171726407
5/5 [==============================] - 0s 500us/step - loss: 0.1694 - acc: 0.7786
로스는 ? [0.1694071888923645, 0.7786259651184082]
r2스코어는?  0.25647141525569217
5/5 [==============================] - 0s 750us/step - loss: 0.1649 - acc: 0.7615
로스는 ? [0.16486942768096924, 0.7615384459495544]
r2스코어는?  0.2715573011349224
5/5 [==============================] - 0s 500us/step - loss: 0.1734 - acc: 0.7615
로스는 ? [0.1733848750591278, 0.7615384459495544]
r2스코어는?  0.2071957372875527
5/5 [==============================] - 0s 499us/step - loss: 0.1520 - acc: 0.7769
로스는 ? [0.1520136296749115, 0.7769230604171753]
r2스코어는?  0.2959631905400145

최종 평균 로스: 0.461309015750885
최종 평균 r2스코어: 0.23234357918708914
'''