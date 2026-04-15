#35_2에서 가져옴
# x_train, x_test는 reshape
# y_tset, y_train OneHotEncoding

import numpy as np
from tensorflow.keras.datasets import mnist
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, LSTM
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
import time
from sklearn.model_selection import KFold
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
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time

# 1. 데이터 로드 및 전처리
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # MNIST 데이터 로드

# 데이터 정규화 (스케일링)
x_train = x_train / 255.0
x_test = x_test / 255.0

# 데이터 Reshape (LSTM을 위한 형태로 변환)
x_train = x_train.reshape(x_train.shape[0], 28 * 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28 * 28, 1)

# One-Hot Encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (60000, 784, 1) (10000, 784, 1) (60000, 10) (10000, 10)

# 2. KFold 설정
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)

# 3. 모델 훈련 및 평가
r2_scores = []
losses = []

for train_index, test_index in kfold.split(x_train):
    x_train1, x_val1 = x_train[train_index], x_train[test_index]
    y_train1, y_val1 = y_train[train_index], y_train[test_index]

    model = Sequential()
    model.add(LSTM(10, input_shape=(28 * 28, 1)))  # timesteps, features
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))  # 다중 클래스 분류

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1, restore_best_weights=True)
    mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True,
                          filepath="./_save/keras_mnist_model.h5")

    start = time.time()
    hist = model.fit(x_train1, y_train1, epochs=100, batch_size=128, verbose=1, validation_data=(x_val1, y_val1),
                     callbacks=[es, mcp])
    end = time.time()

    # 평가 및 예측
    loss = model.evaluate(x_test, y_test)
    losses.append(loss[0])  # 로스값 저장
    y_predict = model.predict(x_test)

    # R2 스코어 계산 (회귀가 아닌 분류이므로 사용하지 않는 것이 좋음)
    r2 = r2_score(y_test.argmax(axis=1), y_predict.argmax(axis=1))  # 다중 클래스 분류에서 R2는 의미가 적음
    r2_scores.append(r2)
    
    print("로스는 ?", loss[0])
    print("정확도는 ?", loss[1])
    print("R2 스코어는? ", r2)

# KFold의 평균 결과 출력
print(f"\n최종 평균 로스: {np.mean(losses)}")
print(f"최종 평균 R2 스코어: {np.mean(r2_scores)}")  # R2 대신 정확도를 사용하는 것이 좋음
