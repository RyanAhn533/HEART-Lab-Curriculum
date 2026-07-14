# ============================================
# 4-5. 이진분류 — Breast Cancer (Sigmoid + BCE)
# 이전(4-4)과 차이: #3에 sigmoid + loss='binary_crossentropy'
#
# 왜 배우는가:
#   회귀(숫자 예측) → 이진분류(0 or 1).
#   2-5 로지스틱 회귀를 DNN으로 확장.
#
# 나중에 만나는 곳:
#   → Chapter 5 CNN (예정, 손계산은 worksheets/): 이미지 이진분류에 동일 구조
#
# ▶ 보고 오기: Coursera C2W1 Lab02 "CoffeeRoasting"
#
# Ref: AI5 keras20~21
# ============================================

#0. 라이브러리 ──────────────────────────────
from tensorflow.keras.models import Sequential     # Sequential: 레이어를 순서대로 쌓는 모델 구조
from tensorflow.keras.layers import Dense           # Dense: 완전 연결 레이어 (모든 뉴런이 연결)
from sklearn.model_selection import train_test_split  # 데이터를 학습/테스트로 분할
from sklearn.preprocessing import StandardScaler    # 데이터 정규화 (평균=0, 표준편차=1)
from sklearn.metrics import accuracy_score, classification_report  # accuracy_score: 정확도, classification_report: 정밀도/재현율 등 상세 보고서
from sklearn.datasets import load_breast_cancer     # 유방암 데이터셋 (sklearn 내장, 양성/악성 분류)
import numpy as np                                  # numpy: 수학 연산 라이브러리

#1. 데이터 ─────────────────────────────────
dataset = load_breast_cancer()                      # 유방암 데이터 로드 (569개 샘플)
x = dataset.data        # (569, 30)                 # 입력: 569명, 30개 특성 (종양 크기, 모양, 질감 등)
y = dataset.target       # (569,)  — 0: 악성, 1: 양성  # 출력: 0=악성(암), 1=양성(정상)

print(f"[ Breast Cancer — 이진분류 ]")
print(f"x shape: {x.shape}, y shape: {y.shape}")    # 데이터 형태 확인
print(f"클래스 분포: 0(악성)={sum(y==0)}, 1(양성)={sum(y==1)}")  # 각 클래스 개수 출력

#2. 데이터 전처리 및 분할 ───────────────────
x_train, x_test, y_train, y_test = train_test_split(  # 80% 학습, 20% 테스트
    x, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()                           # 스케일러 생성
x_train = scaler.fit_transform(x_train)             # train 기준으로 정규화 (fit+transform)
x_test = scaler.transform(x_test)                   # test는 train 기준으로 변환만

#3. 모델 ──────────────────────────────────
model = Sequential()                                # 빈 모델 생성
model.add(Dense(64, activation='relu', input_shape=(x_train.shape[1],)))  # 은닉층1: 뉴런 64개, 입력=30개 특성
model.add(Dense(32, activation='relu'))              # 은닉층2: 뉴런 32개, ReLU 활성화
model.add(Dense(1, activation='sigmoid'))           # ← NEW: 출력층 sigmoid → 0~1 확률 출력 (이진분류)

model.compile(
    loss='binary_crossentropy',                      # ← NEW: BCE 손실함수. 이진분류 전용 (0/1 예측 오차 측정)
    optimizer='adam',                                # Adam 최적화기 (학습률 자동 조정)
    metrics=['accuracy'],                            # ← NEW: 학습 중 정확도도 함께 추적
)

model.fit(x_train, y_train, epochs=50, batch_size=16, verbose=0)  # 50번 반복, 16개씩 학습

#4. 평가 ──────────────────────────────────
loss, acc = model.evaluate(x_test, y_test, verbose=0)  # 테스트 데이터로 손실과 정확도 계산
y_pred_prob = model.predict(x_test, verbose=0).ravel()  # 예측 확률 (0~1 사이 값)
y_pred = (y_pred_prob >= 0.5).astype(int)    # threshold 0.5: 0.5 이상이면 1(양성), 미만이면 0(악성)

print(f"\n[ 평가 ]")
print(f"Loss (BCE): {loss:.4f}")                     # BCE 손실값 (낮을수록 좋음)
print(f"Accuracy: {acc:.4f}")                         # 정확도 (1에 가까울수록 좋음)
print(f"\n{classification_report(y_test, y_pred, target_names=['Malignant', 'Benign'])}")  # 클래스별 상세 성능

print(f"예측 확률 (처음 5개): {y_pred_prob[:5].round(3)}")  # sigmoid 출력값 (0~1 확률)
print(f"예측 클래스:          {y_pred[:5]}")           # 0.5 기준으로 변환한 클래스 (0 또는 1)
print(f"실제 클래스:          {y_test[:5]}")           # 실제 정답
