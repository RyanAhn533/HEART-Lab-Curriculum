from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import time as t
from sklearn.metrics import r2_score
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import r2_score, accuracy_score
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
# 데이터셋 불러오기
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

# 타겟 값을 2차원 배열로 변환 (원-핫 인코딩을 위해 필요)
y = y.reshape(-1, 1)

# OneHotEncoder 초기화 및 타겟 값에 적용
ohe = OneHotEncoder(sparse=False)
y_encoded = ohe.fit_transform(y)

# 결과 출력
print(y_encoded)
print(y_encoded.shape)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7,
                                                    random_state=7777, 
                                                    shuffle=True,
                                                    stratify=y)
#stratify = 0.7 이렇게하면 한쪽에 치우칠 수 있는 것을 막아줌

"""
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
print(pd.DataFrame(y_train).value_counts())
"""

#모델
model = Sequential()
model.add(Dense(32, input_dim=54, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='softmax'))

#컴파일 훈련
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['acc'])
es= EarlyStopping(monitor='val_loss', mode = 'min', patience=500,
                  restore_best_weights=True)

model.fit(x_train, y_train, epochs=1000, batch_size=2048,
          verbose=1, validation_split=0.2, callbacks=[es])

#평가예측
loss = model.evaluate(x_test, y_test)
print("로스는 ?", loss)
y_predict = model.predict(x_test)
y_pred = np.round(y_predict)
accuracy_score = accuracy_score(y_test, y_pred)
r2 = r2_score(y_test, y_predict)
print("r2스코어는? ", r2)
print(y_pred)
print("acc_score : '",accuracy_score)
