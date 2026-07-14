# ============================================
# 4-3. TF2/Keras 첫 코드 — 회귀
# 이전과 차이: Chapter 2~3 sklearn → 처음으로 TF2/Keras 사용
#
# 왜 배우는가:
#   #0→#1→#2→#3→#4 구조를 처음 익힌다.
#   Dense(1) = 2-2 선형회귀와 같은 구조.
#   이 뼈대가 이후 모든 코드의 기반.
#
# 나중에 만나는 곳:
#   → 4-4~4-8: 이 뼈대에서 한 섹션만 바뀜
#   → 후속 챕터 전체: 동일 구조
#
# ▶ 보고 오기: 3B1B "Neural Networks" Ch.1~2
#
# Ref: Coursera C2W1 Lab01 / AI5 keras01~04
# ============================================

#0. 라이브러리 ──────────────────────────────
from tensorflow.keras.models import Sequential    # Sequential: 레이어를 순서대로 쌓는 모델 구조 (가장 기본적인 모델)
from tensorflow.keras.layers import Dense          # Dense: 완전 연결 레이어 (모든 뉴런이 이전 레이어와 연결)
from sklearn.model_selection import train_test_split  # train_test_split: 데이터를 학습용/테스트용으로 분할
from sklearn.preprocessing import StandardScaler   # StandardScaler: 데이터를 평균=0, 표준편차=1로 정규화 (학습 안정화)
from sklearn.metrics import r2_score               # r2_score: 회귀 모델 성능 지표 (1에 가까울수록 좋음)
import numpy as np                                 # numpy: 수학 연산 라이브러리 (배열, 행렬 계산)

#1. 데이터 ─────────────────────────────────
# 간단한 데이터: y = 3x + 2 (+ noise)
np.random.seed(42)                                 # 랜덤 시드 고정 → 실행할 때마다 같은 데이터 생성
x = np.random.rand(200, 1) * 10           # (200, 1) — 0~10 사이 랜덤 숫자 200개 (입력 데이터)
y = 3 * x.ravel() + 2 + np.random.randn(200) * 2  # y = 3x + 2 + 노이즈. ravel()은 2D→1D 변환

print("[ 데이터 ]")
print(f"x shape: {x.shape}, y shape: {y.shape}")   # x: (200,1), y: (200,) 형태 확인
print(f"x 범위: {x.min():.1f} ~ {x.max():.1f}")    # 입력값 범위 출력
print(f"y 범위: {y.min():.1f} ~ {y.max():.1f}")    # 출력값 범위 출력

#2. 데이터 전처리 및 분할 ───────────────────
x_train, x_test, y_train, y_test = train_test_split(  # 80% 학습용, 20% 테스트용으로 분할
    x, y, test_size=0.2, random_state=42           # test_size=0.2: 20%를 테스트로, random_state: 재현성
)

scaler = StandardScaler()                          # 스케일러 객체 생성
x_train = scaler.fit_transform(x_train)    # train으로 평균/표준편차 계산(fit) + 변환(transform) 동시에
x_test = scaler.transform(x_test)          # test는 train 기준으로 변환만 (fit 하면 데이터 누수!)

print(f"\ntrain: {x_train.shape}, test: {x_test.shape}")  # 분할 결과 확인

#3. 모델 ──────────────────────────────────
model = Sequential()                               # 빈 모델 생성 (레이어를 순서대로 추가할 준비)
model.add(Dense(64, activation='relu', input_shape=(1,)))   # 은닉층1: 뉴런 64개, ReLU 활성화, 입력 1개
model.add(Dense(32, activation='relu'))                      # 은닉층2: 뉴런 32개, ReLU 활성화
model.add(Dense(1))                                          # 출력층: 뉴런 1개, 활성화 없음 (회귀=숫자 예측)

model.compile(
    loss='mse',             # ← 손실함수: MSE(평균제곱오차). 예측과 실제의 차이를 제곱해서 평균
    optimizer='adam',       # ← 최적화기: Adam (경사하강법의 개선 버전, 학습률 자동 조정)
)

print("\n[ 모델 구조 ]")
model.summary()                                    # 모델 구조 출력 (레이어, 파라미터 수 등)

history = model.fit(                               # 학습 시작! 데이터를 반복해서 학습
    x_train, y_train,                              # 학습 데이터 (입력, 정답)
    epochs=100,                                    # epochs=100: 전체 데이터를 100번 반복 학습
    batch_size=16,                                 # batch_size=16: 한번에 16개씩 묶어서 학습 (메모리 효율)
    verbose=0,              # 출력 끄기 (깔끔하게)  # verbose=0: 학습 진행 로그 숨김
)

#4. 평가 ──────────────────────────────────
loss = model.evaluate(x_test, y_test, verbose=0)   # 테스트 데이터로 손실(MSE) 계산
y_pred = model.predict(x_test, verbose=0).ravel()   # 테스트 데이터로 예측값 생성. ravel()로 1D 변환
r2 = r2_score(y_test, y_pred)                       # R² 점수 계산 (1=완벽, 0=평균만큼, 음수=평균보다 못함)

print(f"\n[ 평가 ]")
print(f"Test Loss (MSE): {loss:.4f}")               # MSE가 낮을수록 좋음
print(f"R² Score: {r2:.4f}")                         # 1에 가까울수록 좋음
print(f"RMSE: {np.sqrt(loss):.4f}")                  # RMSE = MSE의 제곱근. 원래 단위로 해석 가능

# 예측 확인
print(f"\n예측 vs 실제 (처음 5개):")
for i in range(5):                                   # 처음 5개 샘플의 예측/실제/차이 출력
    print(f"  예측: {y_pred[i]:.2f}, 실제: {y_test[i]:.2f}, 차이: {abs(y_pred[i]-y_test[i]):.2f}")

# ── 시각화 ────────────────────────────────
import matplotlib.pyplot as plt                      # matplotlib: 그래프 그리기 라이브러리

fig, axes = plt.subplots(1, 3, figsize=(14, 4))     # 1행 3열 = 3개 그래프, 크기 14x4인치

# 학습 곡선
axes[0].plot(history.history['loss'], 'b-')          # epoch별 학습 손실(MSE) 변화를 파란 선으로
axes[0].set_title('Training Loss (MSE)')             # 제목: 학습 손실
axes[0].set_xlabel('Epoch')                          # x축: 에폭 (반복 횟수)
axes[0].set_ylabel('Loss')                           # y축: 손실값 (낮을수록 좋음)

# 예측 vs 실제
axes[1].scatter(y_test, y_pred, alpha=0.5, s=20)     # 실제값(x) vs 예측값(y) 산점도
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # 대각선=완벽한 예측
axes[1].set_title(f'Actual vs Predicted (R²={r2:.3f})')  # 제목에 R² 점수 표시
axes[1].set_xlabel('Actual')                         # x축: 실제값
axes[1].set_ylabel('Predicted')                      # y축: 예측값

# 잔차 분포
residuals = y_test - y_pred                          # 잔차 = 실제 - 예측 (오차)
axes[2].hist(residuals, bins=15, edgecolor='black', alpha=0.7)  # 잔차의 히스토그램 (분포 확인)
axes[2].axvline(0, color='red', linestyle='--')      # 0 기준선 (잔차가 0 근처에 몰리면 좋은 모델)
axes[2].set_title('Residual Distribution')           # 제목: 잔차 분포
axes[2].set_xlabel('Residual')                       # x축: 잔차 크기

plt.tight_layout()                                   # 그래프 간 겹침 방지
plt.savefig('4-3_output.png', dpi=100)               # 결과를 PNG로 저장
plt.show()                                           # 화면에 표시

# ── 정리 ──────────────────────────────────
print("\n" + "="*50)
print("핵심 정리:")
print("  #0 라이브러리: tensorflow.keras + sklearn")
print("  #1 데이터: numpy 배열")
print("  #2 전처리: split + StandardScaler")
print("  #3 모델: Sequential → Dense → compile → fit")
print("  #4 평가: evaluate + predict + r2_score")
print("")
print("  Dense(1) = 선형회귀와 같은 구조")
print("  loss='mse' = 2-3")
print("  optimizer='adam' = 2-4")
print("  → 이 뼈대가 이후 모든 코드의 기반!")
print("="*50)
