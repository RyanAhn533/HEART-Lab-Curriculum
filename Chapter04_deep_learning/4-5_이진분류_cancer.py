# ============================================
# 2-5. 이진분류 — Breast Cancer (Sigmoid + BCE)
# 이전(2-4)과 차이: #3에 sigmoid + loss='binary_crossentropy'
#
# 왜 배우는가:
#   회귀(숫자 예측) → 이진분류(0 or 1).
#   Phase 1-10 로지스틱 회귀를 DNN으로 확장.
#
# 나중에 만나는 곳:
#   → Phase 7 CNN: 이미지 이진분류에 동일 구조
#
# ▶ 보고 오기: Coursera C2W1 Lab02 "CoffeeRoasting"
#
# Ref: AI5 keras20~21
# ============================================

#0. 라이브러리 ──────────────────────────────
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_breast_cancer
import numpy as np

#1. 데이터 ─────────────────────────────────
dataset = load_breast_cancer()
x = dataset.data        # (569, 30)
y = dataset.target       # (569,)  — 0: 악성, 1: 양성

print(f"[ Breast Cancer — 이진분류 ]")
print(f"x shape: {x.shape}, y shape: {y.shape}")
print(f"클래스 분포: 0(악성)={sum(y==0)}, 1(양성)={sum(y==1)}")

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
model.add(Dense(1, activation='sigmoid'))           # ← NEW: 출력 sigmoid (0~1)

model.compile(
    loss='binary_crossentropy',                      # ← NEW: BCE (Phase 1-11)
    optimizer='adam',
    metrics=['accuracy'],                            # ← NEW: 정확도도 추적
)

model.fit(x_train, y_train, epochs=50, batch_size=16, verbose=0)

#4. 평가 ──────────────────────────────────
loss, acc = model.evaluate(x_test, y_test, verbose=0)
y_pred_prob = model.predict(x_test, verbose=0).ravel()
y_pred = (y_pred_prob >= 0.5).astype(int)    # threshold 0.5

print(f"\n[ 평가 ]")
print(f"Loss (BCE): {loss:.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Malignant', 'Benign'])}")

print(f"예측 확률 (처음 5개): {y_pred_prob[:5].round(3)}")
print(f"예측 클래스:          {y_pred[:5]}")
print(f"실제 클래스:          {y_test[:5]}")
