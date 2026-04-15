# ============================================
# 2-6. 다중분류 — MNIST (10 클래스, 이미지)
# 이전(2-6 iris)과 차이: #1 데이터셋만 교체 + #2에 reshape
#
# 왜 배우는가:
#   같은 다중분류 구조가 이미지에서도 동작하는지 확인.
#   Phase 7 CNN 전에 DNN으로 먼저 해본다.
#
# Ref: Coursera C2W2 Assignment / AI5 keras22
# ============================================

#0. 라이브러리 ──────────────────────────────
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from sklearn.metrics import accuracy_score
import numpy as np

#1. 데이터 ─────────────────────────────────
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f"[ MNIST — 손글씨 숫자 (10 클래스) ]")
print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
print(f"x_test: {x_test.shape}, y_test: {y_test.shape}")
print(f"픽셀 범위: {x_train.min()} ~ {x_train.max()}")

#2. 데이터 전처리 ──────────────────────────
# 정규화 (0~255 → 0~1)
x_train = x_train / 255.0                                   # ← NEW: 이미지 정규화
x_test = x_test / 255.0

# Flatten (28x28 → 784) — DNN은 1D 입력만 받음
x_train = x_train.reshape(x_train.shape[0], -1)             # (60000, 784)
x_test = x_test.reshape(x_test.shape[0], -1)                # (10000, 784)

# One-Hot
y_train_oh = to_categorical(y_train, 10)
y_test_oh = to_categorical(y_test, 10)

#3. 모델 ──────────────────────────────────
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(784,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))                   # 10개 클래스

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'],
)

model.fit(x_train, y_train_oh, epochs=10, batch_size=32, verbose=1)

#4. 평가 ──────────────────────────────────
loss, acc = model.evaluate(x_test, y_test_oh, verbose=0)
y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)

print(f"\n[ 평가 ]")
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"\n→ DNN만으로도 MNIST {acc:.1%} 정확도")
print(f"→ Phase 7 CNN으로 바꾸면 더 좋아진다")
