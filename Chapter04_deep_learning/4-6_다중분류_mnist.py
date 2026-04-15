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
from tensorflow.keras.models import Sequential     # Sequential: 레이어를 순서대로 쌓는 모델 구조
from tensorflow.keras.layers import Dense, Flatten  # Dense: 완전 연결 레이어, Flatten: 다차원→1차원 변환 (여기선 미사용)
from tensorflow.keras.utils import to_categorical   # to_categorical: 정수 라벨→원핫 벡터 변환
from tensorflow.keras.datasets import mnist         # MNIST: 손글씨 숫자 이미지 데이터셋 (0~9, 28x28 픽셀)
from sklearn.metrics import accuracy_score          # accuracy_score: 정확도 계산
import numpy as np                                  # numpy: 수학 연산 라이브러리

#1. 데이터 ─────────────────────────────────
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # MNIST 데이터 로드 (자동으로 학습/테스트 분리됨)

print(f"[ MNIST — 손글씨 숫자 (10 클래스) ]")
print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")  # x: (60000, 28, 28) = 6만장 28x28 이미지
print(f"x_test: {x_test.shape}, y_test: {y_test.shape}")      # 테스트: 1만장
print(f"픽셀 범위: {x_train.min()} ~ {x_train.max()}")        # 0~255 (흑백 밝기)

#2. 데이터 전처리 ──────────────────────────
# 정규화 (0~255 → 0~1)
x_train = x_train / 255.0                                   # ← NEW: 픽셀값을 0~1로 스케일링 (학습 안정화)
x_test = x_test / 255.0                                     # 테스트도 같은 방식으로 정규화

# Flatten (28x28 → 784) — DNN은 1D 입력만 받음
x_train = x_train.reshape(x_train.shape[0], -1)             # (60000, 28, 28) → (60000, 784) 2D를 1D로 펼침
x_test = x_test.reshape(x_test.shape[0], -1)                # (10000, 28, 28) → (10000, 784)

# One-Hot
y_train_oh = to_categorical(y_train, 10)                    # 정수(0~9) → 원핫 벡터 (예: 3→[0,0,0,1,0,0,0,0,0,0])
y_test_oh = to_categorical(y_test, 10)                      # 10개 클래스이므로 10차원 벡터

#3. 모델 ──────────────────────────────────
model = Sequential()                                        # 빈 모델 생성
model.add(Dense(128, activation='relu', input_shape=(784,)))  # 은닉층1: 뉴런 128개, 입력=784(28x28 펼친 것)
model.add(Dense(64, activation='relu'))                       # 은닉층2: 뉴런 64개
model.add(Dense(10, activation='softmax'))                   # 출력층: 10개 뉴런 + softmax (0~9 각 숫자의 확률)

model.compile(
    loss='categorical_crossentropy',                         # CCE: 다중분류 손실함수 (원핫 라벨과 비교)
    optimizer='adam',                                        # Adam 최적화기
    metrics=['accuracy'],                                    # 학습 중 정확도 추적
)

model.fit(x_train, y_train_oh, epochs=10, batch_size=32, verbose=1)  # 10번 반복, 32개씩 학습 (verbose=1: 진행상황 표시)

#4. 평가 ──────────────────────────────────
loss, acc = model.evaluate(x_test, y_test_oh, verbose=0)     # 테스트 데이터로 손실과 정확도 계산
y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)  # softmax 확률 → 가장 높은 확률의 숫자 선택

print(f"\n[ 평가 ]")
print(f"Loss: {loss:.4f}")                                    # 손실값
print(f"Accuracy: {acc:.4f}")                                  # 정확도
print(f"\n→ DNN만으로도 MNIST {acc:.1%} 정확도")              # DNN으로도 꽤 높은 정확도 달성
print(f"→ Phase 7 CNN으로 바꾸면 더 좋아진다")                 # CNN이 이미지 분류에 더 적합
