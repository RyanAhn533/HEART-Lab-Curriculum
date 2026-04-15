from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import numpy as np
import time as t
from tensorflow.keras.callbacks import EarlyStopping
##############
# 데이터 로드 및 전처리
dataset = load_boston()
x = dataset.data
y = dataset.target

# 데이터 스케일링
scaler = StandardScaler()
x = scaler.fit_transform(x)

# 훈련 및 테스트 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=3)

# 모델 구성
model = Sequential()
model.add(Dense(64, input_dim=13, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))  # 회귀 문제이므로 출력 노드는 1개

# 모델 컴파일 및 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=1000, batch_size=32, verbose=2, validation_split=0.3, callbacks=[es])

# 평가 및 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

# R2 스코어 계산
r2 = r2_score(y_test, y_predict)
print("로스 :", loss)
print("R2 스코어:", r2)
