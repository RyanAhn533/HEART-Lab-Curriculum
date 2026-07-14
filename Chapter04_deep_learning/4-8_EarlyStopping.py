# ============================================
# 4-8. validation + EarlyStopping + 학습곡선
# 이전(4-5)과 차이: #3에 validation_split + EarlyStopping 콜백 추가
#
# 왜 배우는가:
#   2-8 과적합을 코드로 감지하고 방지.
#   실무에서 model.fit()에 거의 항상 들어가는 것.
#
# 나중에 만나는 곳:
#   → 모든 후속 챕터의 #3: 콜백 필수
#
# ▶ 보고 오기: Coursera C2W3 "Advice for ML"
#
# Ref: AI5 keras15~19
# ============================================

#0. 라이브러리 ──────────────────────────────
from tensorflow.keras.models import Sequential     # Sequential: 레이어를 순서대로 쌓는 모델 구조
from tensorflow.keras.layers import Dense           # Dense: 완전 연결 레이어
from tensorflow.keras.callbacks import EarlyStopping       # ← NEW: 과적합 방지 콜백 (검증 손실 감시 → 자동 학습 중단)
from sklearn.model_selection import train_test_split  # 데이터를 학습/테스트로 분할
from sklearn.preprocessing import StandardScaler    # 데이터 정규화 (평균=0, 표준편차=1)
from sklearn.metrics import r2_score                # R² 점수: 회귀 성능 지표
from sklearn.datasets import fetch_california_housing  # 캘리포니아 집값 데이터셋
import numpy as np                                  # numpy: 수학 연산 라이브러리
import matplotlib.pyplot as plt                     # matplotlib: 그래프 시각화 라이브러리

#1. 데이터 ─────────────────────────────────
dataset = fetch_california_housing()                # 캘리포니아 집값 데이터 로드
x = dataset.data                                    # 입력 데이터 (20640개, 8개 특성)
y = dataset.target                                  # 출력 데이터 (집값)

#2. 데이터 전처리 및 분할 ───────────────────
x_train, x_test, y_train, y_test = train_test_split(  # 80% 학습, 20% 테스트
    x, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()                           # 스케일러 생성
x_train = scaler.fit_transform(x_train)             # train 기준 정규화 (fit=기준 계산 + transform=변환)
x_test = scaler.transform(x_test)                   # test는 train 기준으로 변환만 (데이터 누수 방지)

#3. 모델 ──────────────────────────────────
model = Sequential()                                # 빈 모델 생성
model.add(Dense(128, activation='relu', input_shape=(x_train.shape[1],)))  # 은닉층1: 뉴런 128개, 입력=8개 특성
model.add(Dense(64, activation='relu'))              # 은닉층2: 뉴런 64개
model.add(Dense(32, activation='relu'))              # 은닉층3: 뉴런 32개 (더 깊은 모델 → 과적합 가능성↑)
model.add(Dense(1))                                  # 출력층: 뉴런 1개, 활성화 없음 (회귀)

model.compile(loss='mse', optimizer='adam')           # 손실=MSE, 최적화=Adam

# ── EarlyStopping 콜백 ──────────────────  ← NEW
# 과적합 방지: 검증 손실이 더 이상 개선되지 않으면 학습을 자동으로 중단
es = EarlyStopping(
    monitor='val_loss',           # 감시 대상: 검증 손실 (val_loss). 이 값이 줄어들지 않으면 멈춤
    patience=20,                  # 참을성: 20 epoch 연속 개선 없으면 멈춤 (너무 작으면 일찍 멈춤)
    restore_best_weights=True,    # 가장 좋았던 시점의 가중치로 복원 (마지막이 아닌 최적 상태)
    verbose=1,                    # 멈출 때 메시지 출력
)

history = model.fit(
    x_train, y_train,                                # 학습 데이터
    epochs=500,                   # 크게 잡아도 ES가 알아서 멈춤 (최대 500번이지만 보통 훨씬 적게 학습)
    batch_size=32,                                   # 32개씩 묶어서 학습
    validation_split=0.2,         # ← NEW: 학습 데이터의 20%를 검증용으로 분리 (과적합 감시용)
    callbacks=[es],               # ← NEW: EarlyStopping 콜백 전달 (학습 중 매 epoch마다 실행)
    verbose=0,                                       # 학습 로그 숨김
)

# 실제 학습한 epoch 수
actual_epochs = len(history.history['loss'])          # EarlyStopping으로 실제 학습된 횟수 확인
print(f"\n설정 epochs: 500, 실제 학습: {actual_epochs} epoch (EarlyStopping)")  # 500보다 훨씬 적을 것

#4. 평가 ──────────────────────────────────
loss = model.evaluate(x_test, y_test, verbose=0)     # 테스트 데이터로 MSE 계산
y_pred = model.predict(x_test, verbose=0).ravel()     # 예측값 생성
r2 = r2_score(y_test, y_pred)                         # R² 점수 계산

print(f"\n[ 평가 ]")
print(f"Test MSE: {loss:.4f}")                        # 테스트 손실
print(f"R²: {r2:.4f}")                                # R² (1에 가까울수록 좋음)

# ── 학습곡선 시각화 ──────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))      # 3개 그래프 캔버스

# Train vs Val Loss — 과적합 감지의 핵심 그래프
best_epoch = int(np.argmin(history.history['val_loss']))            # restore_best_weights가 복원하는 실제 최적 시점 = val_loss 최소 epoch
axes[0].plot(history.history['loss'], 'b-', label='Train Loss')    # 학습 손실 (파랑)
axes[0].plot(history.history['val_loss'], 'r-', label='Val Loss')  # 검증 손실 (빨강) — train보다 높으면 과적합
axes[0].axvline(best_epoch, color='green', linestyle='--', alpha=0.5, label=f'Best Epoch={best_epoch}')  # 최적 시점 표시 (np.argmin으로 실측)
axes[0].set_title('Learning Curve')                   # 제목: 학습 곡선
axes[0].set_xlabel('Epoch')                           # x축: 에폭
axes[0].set_ylabel('Loss (MSE)')                      # y축: 손실
axes[0].legend()                                      # 범례 표시

# 과적합 감지 영역 — val과 train의 차이
train_loss = history.history['loss']                  # 학습 손실 기록
val_loss = history.history['val_loss']                # 검증 손실 기록
gap = [v - t for t, v in zip(train_loss, val_loss)]   # gap = 검증손실 - 학습손실 (양수면 과적합 신호)
axes[1].plot(gap, 'orange', linewidth=2)              # gap 변화를 주황색으로
axes[1].axhline(0, color='gray', linestyle='--')      # 0 기준선
axes[1].fill_between(range(len(gap)), gap, 0, where=[g > 0 for g in gap], alpha=0.2, color='red')  # 과적합 영역 빨간색
axes[1].set_title('Val - Train Gap (Overfitting Signal)')  # 제목: 과적합 신호
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Gap')                             # gap이 커지면 과적합 심화

# 예측 vs 실제
axes[2].scatter(y_test, y_pred, alpha=0.3, s=5)       # 실제(x) vs 예측(y) 산점도
axes[2].plot([0, 5], [0, 5], 'r--')                   # 대각선 = 완벽한 예측 기준선
axes[2].set_title(f'Actual vs Predicted (R²={r2:.3f})')  # R² 점수 표시
axes[2].set_xlabel('Actual')                          # x축: 실제값
axes[2].set_ylabel('Predicted')                       # y축: 예측값

plt.tight_layout()                                    # 그래프 간 겹침 방지
plt.savefig('4-8_output.png', dpi=100)                # PNG로 저장
plt.show()                                            # 화면에 표시

# ── 정리 ──────────────────────────────────
print("\n" + "="*50)
print("핵심 정리:")
print("  validation_split=0.2 → 검증 데이터 20% 분리")
print("  EarlyStopping → val_loss 감시, 자동 멈춤")
print("  restore_best_weights → 최적 가중치 복원")
print("  학습곡선 → train↓ val↑ = 과적합!")
print("  → 이후 모든 코드에 ES가 들어간다")
print("="*50)
print("\n★ Chapter 4 완료! → 체크포인트 시험 2")
