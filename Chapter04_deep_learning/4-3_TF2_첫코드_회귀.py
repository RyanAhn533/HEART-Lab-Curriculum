# ============================================
# 2-3. TF2/Keras 첫 코드 — 회귀
# 이전과 차이: Phase 1 sklearn → 처음으로 TF2/Keras 사용
#
# 왜 배우는가:
#   #0→#1→#2→#3→#4 구조를 처음 익힌다.
#   Dense(1) = Phase 1-7 선형회귀와 같은 구조.
#   이 뼈대가 이후 모든 코드의 기반.
#
# 나중에 만나는 곳:
#   → 2-4~2-8: 이 뼈대에서 한 섹션만 바뀜
#   → Phase 3 이후 전체: 동일 구조
#
# ▶ 보고 오기: 3B1B "Neural Networks" Ch.1~2
#
# Ref: Coursera C2W1 Lab01 / AI5 keras01~04
# ============================================

#0. 라이브러리 ──────────────────────────────
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import numpy as np

#1. 데이터 ─────────────────────────────────
# 간단한 데이터: y = 3x + 2 (+ noise)
np.random.seed(42)
x = np.random.rand(200, 1) * 10           # (200, 1) — 0~10
y = 3 * x.ravel() + 2 + np.random.randn(200) * 2  # y = 3x + 2 + noise

print("[ 데이터 ]")
print(f"x shape: {x.shape}, y shape: {y.shape}")
print(f"x 범위: {x.min():.1f} ~ {x.max():.1f}")
print(f"y 범위: {y.min():.1f} ~ {y.max():.1f}")

#2. 데이터 전처리 및 분할 ───────────────────
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)    # train 기준으로 fit
x_test = scaler.transform(x_test)          # test는 transform만

print(f"\ntrain: {x_train.shape}, test: {x_test.shape}")

#3. 모델 ──────────────────────────────────
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(1,)))   # 은닉층: ReLU
model.add(Dense(32, activation='relu'))                      # 은닉층: ReLU
model.add(Dense(1))                                          # 출력층: 활성화 없음 (회귀)

model.compile(
    loss='mse',             # ← Phase 1-8에서 배운 MSE
    optimizer='adam',       # ← Phase 1-9에서 배운 경사하강법
)

print("\n[ 모델 구조 ]")
model.summary()

history = model.fit(
    x_train, y_train,
    epochs=100,
    batch_size=16,
    verbose=0,              # 출력 끄기 (깔끔하게)
)

#4. 평가 ──────────────────────────────────
loss = model.evaluate(x_test, y_test, verbose=0)
y_pred = model.predict(x_test, verbose=0).ravel()
r2 = r2_score(y_test, y_pred)

print(f"\n[ 평가 ]")
print(f"Test Loss (MSE): {loss:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {np.sqrt(loss):.4f}")

# 예측 확인
print(f"\n예측 vs 실제 (처음 5개):")
for i in range(5):
    print(f"  예측: {y_pred[i]:.2f}, 실제: {y_test[i]:.2f}, 차이: {abs(y_pred[i]-y_test[i]):.2f}")

# ── 시각화 ────────────────────────────────
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# 학습 곡선
axes[0].plot(history.history['loss'], 'b-')
axes[0].set_title('Training Loss (MSE)')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')

# 예측 vs 실제
axes[1].scatter(y_test, y_pred, alpha=0.5, s=20)
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[1].set_title(f'Actual vs Predicted (R²={r2:.3f})')
axes[1].set_xlabel('Actual')
axes[1].set_ylabel('Predicted')

# 잔차 분포
residuals = y_test - y_pred
axes[2].hist(residuals, bins=15, edgecolor='black', alpha=0.7)
axes[2].axvline(0, color='red', linestyle='--')
axes[2].set_title('Residual Distribution')
axes[2].set_xlabel('Residual')

plt.tight_layout()
plt.savefig('2-3_output.png', dpi=100)
plt.show()

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
print("  loss='mse' = Phase 1-8")
print("  optimizer='adam' = Phase 1-9")
print("  → 이 뼈대가 이후 모든 코드의 기반!")
print("="*50)
