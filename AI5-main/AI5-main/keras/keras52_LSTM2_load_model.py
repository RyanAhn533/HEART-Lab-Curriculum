import numpy as np
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Input, BatchNormalization
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
import time
from sklearn.model_selection import train_test_split
"""
#1. Data
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],
             [7,8,9],[8,9,10],[9,10,11],[10,11,12],[20,30,40],
             [30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50,60,70])

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=3)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

model_path = 'C:\\프로그램\\ai5\\_save\\keras52\\k52_020807_1722_0001-861.3182.hdf5'
model = load_model(model_path)
# 모델 컴파일
model.compile(loss='mse', optimizer='adam', )

#4. 평가, 예측
results = model.evaluate(x,y)
print('loss : ', results)

x_pred = np.array([50,60,70]).reshape(1,3,1) #[[[8]]]
#벡터형태 데이터 (3,) -> (1,3,1)
#스칼라는 행렬아님
y_pred = model.predict(x_pred)
print('[50,60,70]의 결과', y_pred)

"""
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. 데이터 준비
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],
             [7,8,9],[8,9,10],[9,10,11],[10,11,12],[20,30,40],
             [30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50,60,70])

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=3)

# StandardScaler 객체 생성 및 스케일링
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_pred = scaler.transform(x_predict.reshape(1, -1)).reshape(1, 3, 1)  # x_predict 스케일링 및 reshape

# 2. 모델 로드 및 컴파일
model_path = 'C:\\프로그램\\ai5\\_save\\keras52\\k52_020807_1742_0595-0.0040.hdf5'
model = load_model(model_path)

# 모델 컴파일 (이미 학습된 모델이므로 다시 컴파일하지 않아도 됩니다)
model.compile(loss='mse', optimizer='adam')

# 3. 평가
results = model.evaluate(x_test, y_test)
print('loss : ', results)

# 4. 예측
y_pred = model.predict(x_pred)
print('[50,60,70]의 결과', y_pred)
