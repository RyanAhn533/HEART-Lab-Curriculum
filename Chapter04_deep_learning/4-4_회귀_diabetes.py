# ============================================
# 4-4. 회귀 — Diabetes
# 이전(4-4 california)과 차이: #1의 데이터셋만 교체
#
# Ref: AI5 keras11_3_diabetes
# ============================================

#0. 라이브러리 ──────────────────────────────
from tensorflow.keras.models import Sequential     # Sequential: 레이어를 순서대로 쌓는 모델 구조
from tensorflow.keras.layers import Dense           # Dense: 완전 연결 레이어 (모든 뉴런이 연결)
from sklearn.model_selection import train_test_split  # 데이터를 학습/테스트로 분할하는 함수
from sklearn.preprocessing import StandardScaler    # 데이터 정규화 (평균=0, 표준편차=1로 맞춤)
from sklearn.metrics import r2_score                # R² 점수: 회귀 모델 성능 측정 (1=완벽)
from sklearn.datasets import load_diabetes          # 당뇨병 데이터셋 (sklearn 내장, 환자 442명)
import numpy as np                                  # numpy: 수학 연산 라이브러리

#1. 데이터 ─────────────────────────────────  ← 여기만 바뀜!
dataset = load_diabetes()                           # 당뇨병 데이터 로드 (나이, BMI, 혈압 등 → 1년 후 진행도 예측)
x = dataset.data        # (442, 10)                 # 입력: 환자 442명, 10개 특성 (나이, 성별, BMI 등)
y = dataset.target       # (442,)                   # 출력: 1년 후 당뇨 진행 정도 (숫자가 클수록 심함)

print(f"[ Diabetes ]")
print(f"x shape: {x.shape}, y shape: {y.shape}")    # 데이터 형태 확인

#2. 데이터 전처리 및 분할 ───────────────────
x_train, x_test, y_train, y_test = train_test_split(  # 80% 학습, 20% 테스트로 분할
    x, y, test_size=0.2, random_state=42            # random_state=42: 매번 같은 분할 결과 보장
)
scaler = StandardScaler()                           # 스케일러 생성
x_train = scaler.fit_transform(x_train)             # train 기준으로 정규화 (fit=기준 계산, transform=적용)
x_test = scaler.transform(x_test)                   # test는 train 기준으로 변환만 (fit 하면 안 됨!)

#3. 모델 ──────────────────────────────────
model = Sequential()                                # 빈 모델 생성
model.add(Dense(64, activation='relu', input_shape=(x_train.shape[1],)))  # 은닉층1: 뉴런 64개, 입력=특성 수(10)
model.add(Dense(32, activation='relu'))              # 은닉층2: 뉴런 32개, ReLU 활성화 (비선형)
model.add(Dense(1))                                  # 출력층: 뉴런 1개, 활성화 없음 (회귀=숫자 예측)

model.compile(loss='mse', optimizer='adam')           # loss=MSE(오차 제곱 평균), optimizer=Adam(자동 학습률 조정)
model.fit(x_train, y_train, epochs=100, batch_size=16, verbose=0)  # 100번 반복, 16개씩 묶어서 학습

#4. 평가 ──────────────────────────────────
loss = model.evaluate(x_test, y_test, verbose=0)     # 테스트 데이터로 MSE 계산 (낮을수록 좋음)
y_pred = model.predict(x_test, verbose=0).ravel()     # 예측값 생성, ravel()로 1D 배열로 변환
r2 = r2_score(y_test, y_pred)                         # R² 계산 (1에 가까울수록 예측이 정확)

print(f"\n[ 평가 ]")
print(f"MSE: {loss:.4f}, R²: {r2:.4f}")               # 손실과 R² 출력
