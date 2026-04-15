# ============================================
# 2-4. 회귀 — Diabetes
# 이전(2-4 boston)과 차이: #1의 데이터셋만 교체
#
# Ref: AI5 keras11_3_diabetes
# ============================================

#0. 라이브러리 ──────────────────────────────
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes
import numpy as np

#1. 데이터 ─────────────────────────────────  ← 여기만 바뀜!
dataset = load_diabetes()
x = dataset.data        # (442, 10)
y = dataset.target       # (442,)

print(f"[ Diabetes ]")
print(f"x shape: {x.shape}, y shape: {y.shape}")

#2. 데이터 전처리 및 분할 ───────────────────
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#3. 모델 ──────────────────────────────────
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=16, verbose=0)

#4. 평가 ──────────────────────────────────
loss = model.evaluate(x_test, y_test, verbose=0)
y_pred = model.predict(x_test, verbose=0).ravel()
r2 = r2_score(y_test, y_pred)

print(f"\n[ 평가 ]")
print(f"MSE: {loss:.4f}, R²: {r2:.4f}")
