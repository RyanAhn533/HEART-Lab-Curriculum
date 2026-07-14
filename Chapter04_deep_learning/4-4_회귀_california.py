# ============================================
# 4-4. 회귀 — California Housing
# 이전(4-3)과 차이: #1에 실제 데이터셋 적용 (나머지 동일)
#
# 왜 배우는가:
#   4-3의 뼈대가 실제 데이터에서도 그대로 동작하는지 확인.
#   2-2 sklearn 선형회귀와 성능 비교.
#
# ▶ 보고 오기: 구글 "california housing dataset 설명"
#
# Ref: AI5 keras11_1_boston (데이터셋은 California로 교체)
# ============================================

#0. 라이브러리 ──────────────────────────────
from tensorflow.keras.models import Sequential     # Sequential: 레이어를 순서대로 쌓는 모델 구조
from tensorflow.keras.layers import Dense           # Dense: 완전 연결 레이어 (모든 뉴런이 이전 레이어와 연결)
from sklearn.model_selection import train_test_split  # train_test_split: 데이터를 학습/테스트로 분할
from sklearn.preprocessing import StandardScaler    # StandardScaler: 데이터 정규화 (평균=0, 표준편차=1)
from sklearn.metrics import r2_score                # r2_score: 회귀 성능 지표 (1에 가까울수록 좋음)
from sklearn.datasets import fetch_california_housing  # 캘리포니아 집값 데이터셋 (Boston 대체, 윤리적 이슈로)
import numpy as np                                  # numpy: 수학 연산 라이브러리

#1. 데이터 ─────────────────────────────────  ← 여기만 바뀜!
dataset = fetch_california_housing()                # 캘리포니아 집값 데이터 로드 (sklearn 내장)
x = dataset.data        # (20640, 8)                # 입력: 20640개 집, 8개 특성 (위치, 방수, 소득 등)
y = dataset.target       # (20640,)                 # 출력: 집값 (단위: $100,000)

print(f"[ California Housing ]")
print(f"x shape: {x.shape}, y shape: {y.shape}")    # 데이터 형태 확인
print(f"특성: {dataset.feature_names}")              # 8개 특성 이름 출력

#2. 데이터 전처리 및 분할 ───────────────────
x_train, x_test, y_train, y_test = train_test_split(  # 80% 학습, 20% 테스트로 분할
    x, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()                           # 스케일러 객체 생성
x_train = scaler.fit_transform(x_train)             # train으로 기준 계산(fit) + 변환(transform)
x_test = scaler.transform(x_test)                   # test는 train 기준으로 변환만 (데이터 누수 방지)

#3. 모델 ──────────────────────────────────
model = Sequential()                                # 빈 모델 생성
model.add(Dense(64, activation='relu', input_shape=(x_train.shape[1],)))  # 은닉층1: 뉴런 64개, 입력=특성 수(8)
model.add(Dense(32, activation='relu'))              # 은닉층2: 뉴런 32개, ReLU 활성화
model.add(Dense(1))                                  # 출력층: 뉴런 1개, 활성화 없음 (회귀=숫자 예측)

model.compile(loss='mse', optimizer='adam')           # 손실=MSE(평균제곱오차), 최적화=Adam
model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=0)  # 50번 반복 학습, 32개씩 묶어서

#4. 평가 ──────────────────────────────────
loss = model.evaluate(x_test, y_test, verbose=0)     # 테스트 데이터로 MSE 계산
y_pred = model.predict(x_test, verbose=0).ravel()     # 예측값 생성. ravel()로 1D 변환
r2 = r2_score(y_test, y_pred)                         # R² 점수 계산

print(f"\n[ 평가 ]")
print(f"MSE: {loss:.4f}, R²: {r2:.4f}")               # 결과 출력

# 2-2 sklearn과 비교
from sklearn.linear_model import LinearRegression     # LinearRegression: sklearn의 단순 선형회귀 모델
lr = LinearRegression()                               # 선형회귀 모델 생성 (비교용)
lr.fit(x_train, y_train)                              # #2에서 만든 동일한 분할·스케일 데이터 재사용 (재분할/scaler 재fit 금지 — 위에서 배운 원칙)
r2_lr = lr.score(x_test, y_test)                      # 같은 테스트 데이터로 R² 계산 (공정한 비교)
print(f"sklearn LinearRegression R²: {r2_lr:.4f}")    # sklearn 선형회귀 R² 출력
print(f"TF2/Keras DNN R²: {r2:.4f}")                  # 딥러닝 모델 R² 출력
print(f"→ {'DNN이 더 좋음' if r2 > r2_lr else 'Linear가 더 좋음 (데이터가 선형적)'}")  # 비교 결과
