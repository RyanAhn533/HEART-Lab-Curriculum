# ============================================
# 2-4. 회귀 — Boston Housing
# 이전(2-3)과 차이: #1에 실제 데이터셋 적용 (나머지 동일)
#
# 왜 배우는가:
#   2-3의 뼈대가 실제 데이터에서도 그대로 동작하는지 확인.
#   Phase 1-7 sklearn 선형회귀와 성능 비교.
#
# ▶ 보고 오기: 구글 "boston housing dataset 설명"
#
# Ref: AI5 keras11_1_boston
# ============================================

#0. 라이브러리 ──────────────────────────────
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing  # Boston 대체
import numpy as np

#1. 데이터 ─────────────────────────────────  ← 여기만 바뀜!
dataset = fetch_california_housing()
x = dataset.data        # (20640, 8)
y = dataset.target       # (20640,)

print(f"[ California Housing ]")
print(f"x shape: {x.shape}, y shape: {y.shape}")
print(f"특성: {dataset.feature_names}")

#2. 데이터 전처리 및 분할 ───────────────────
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#3. 모델 ──────────────────────────────────
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(x_train.shape[1],)))  # ← input_shape만 바뀜
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=0)

#4. 평가 ──────────────────────────────────
loss = model.evaluate(x_test, y_test, verbose=0)
y_pred = model.predict(x_test, verbose=0).ravel()
r2 = r2_score(y_test, y_pred)

print(f"\n[ 평가 ]")
print(f"MSE: {loss:.4f}, R²: {r2:.4f}")

# Phase 1-7 sklearn과 비교
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(scaler.fit_transform(train_test_split(dataset.data, test_size=0.2, random_state=42)[0]),
       train_test_split(dataset.target, test_size=0.2, random_state=42)[0])
r2_lr = lr.score(scaler.transform(train_test_split(dataset.data, test_size=0.2, random_state=42)[1]),
                 train_test_split(dataset.target, test_size=0.2, random_state=42)[1])
print(f"sklearn LinearRegression R²: {r2_lr:.4f}")
print(f"TF2/Keras DNN R²: {r2:.4f}")
print(f"→ {'DNN이 더 좋음' if r2 > r2_lr else 'Linear가 더 좋음 (데이터가 선형적)'}")
