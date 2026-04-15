# ============================================
# 2-6. 다중분류 — Iris (Softmax + CCE + OneHot)
# 이전(2-5)과 차이: #3에 softmax, #2에 to_categorical
#
# 왜 배우는가:
#   이진(2개) → 다중(3개 이상) 분류.
#   Phase 1-10 softmax를 DNN으로 확장.
#
# ▶ 보고 오기: Coursera C2W2 "Multiclass Classification"
#
# Ref: Coursera C2W2 Multiclass_TF Lab / AI5 keras22~25
# ============================================

#0. 라이브러리 ──────────────────────────────
from tensorflow.keras.models import Sequential     # Sequential: 레이어를 순서대로 쌓는 모델 구조
from tensorflow.keras.layers import Dense           # Dense: 완전 연결 레이어
from tensorflow.keras.utils import to_categorical          # ← NEW: 정수 라벨을 원핫 벡터로 변환 (0→[1,0,0], 1→[0,1,0])
from sklearn.model_selection import train_test_split  # 데이터를 학습/테스트로 분할
from sklearn.preprocessing import StandardScaler    # 데이터 정규화
from sklearn.metrics import accuracy_score, classification_report  # 분류 성능 지표
from sklearn.datasets import load_iris              # 붓꽃 데이터셋 (3종류 분류, 머신러닝의 Hello World)
import numpy as np                                  # numpy: 수학 연산 라이브러리

#1. 데이터 ─────────────────────────────────
dataset = load_iris()                               # 붓꽃 데이터 로드 (150개 샘플, 3종류)
x = dataset.data        # (150, 4)                  # 입력: 150개 꽃, 4개 특성 (꽃잎 길이/폭, 꽃받침 길이/폭)
y = dataset.target       # (150,)  — 0, 1, 2 (3개 클래스)  # 출력: 0=setosa, 1=versicolor, 2=virginica

print(f"[ Iris — 다중분류 (3 클래스) ]")
print(f"x shape: {x.shape}, y shape: {y.shape}")    # 데이터 형태 확인
print(f"클래스: {dataset.target_names}")              # 3개 클래스 이름 출력

#2. 데이터 전처리 및 분할 ───────────────────
x_train, x_test, y_train, y_test = train_test_split(  # 80% 학습, 20% 테스트
    x, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()                           # 스케일러 생성
x_train = scaler.fit_transform(x_train)             # train 기준 정규화
x_test = scaler.transform(x_test)                   # test는 train 기준 변환

# One-Hot Encoding                                         ← NEW
# 정수 라벨(0,1,2)을 원핫 벡터로 변환 (softmax 출력과 비교하기 위해)
# 예: 0→[1,0,0], 1→[0,1,0], 2→[0,0,1]
y_train_oh = to_categorical(y_train, num_classes=3)  # 학습 라벨을 원핫으로 변환
y_test_oh = to_categorical(y_test, num_classes=3)    # 테스트 라벨도 원핫으로 변환

print(f"\nOne-Hot 예시:")
print(f"  원본: {y_train[:3]}")                      # 예: [1, 0, 2]
print(f"  변환: {y_train_oh[:3]}")                   # 예: [[0,1,0], [1,0,0], [0,0,1]]

#3. 모델 ──────────────────────────────────
model = Sequential()                                # 빈 모델 생성
model.add(Dense(64, activation='relu', input_shape=(x_train.shape[1],)))  # 은닉층1: 뉴런 64개, 입력=4개 특성
model.add(Dense(32, activation='relu'))              # 은닉층2: 뉴런 32개
model.add(Dense(3, activation='softmax'))                   # ← NEW: 출력 뉴런 3개 + softmax (3클래스 확률, 합=1)

model.compile(
    loss='categorical_crossentropy',                         # ← NEW: CCE 손실함수. 다중분류 전용 (원핫 라벨과 비교)
    optimizer='adam',                                        # Adam 최적화기
    metrics=['accuracy'],                                    # 정확도 추적
)

model.fit(x_train, y_train_oh, epochs=50, batch_size=8, verbose=0)  # 50번 반복, 8개씩 학습 (원핫 라벨 사용!)

#4. 평가 ──────────────────────────────────
loss, acc = model.evaluate(x_test, y_test_oh, verbose=0)    # 테스트 손실과 정확도
y_pred_prob = model.predict(x_test, verbose=0)               # softmax 출력: 각 클래스별 확률 (3개 값)
y_pred = np.argmax(y_pred_prob, axis=1)     # 확률 → 클래스 번호: 가장 높은 확률의 인덱스 선택

print(f"\n[ 평가 ]")
print(f"Loss (CCE): {loss:.4f}")                     # CCE 손실 (낮을수록 좋음)
print(f"Accuracy: {acc:.4f}")                         # 정확도
print(f"\n{classification_report(y_test, y_pred, target_names=dataset.target_names)}")  # 클래스별 상세

print(f"Softmax 출력 (처음 3개):")
for i in range(3):                                    # softmax 확률과 예측/실제 클래스 비교
    print(f"  {y_pred_prob[i].round(3)} → 클래스 {y_pred[i]} (실제: {y_test[i]})")
