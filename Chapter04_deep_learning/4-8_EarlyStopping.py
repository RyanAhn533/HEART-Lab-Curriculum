# ============================================
# 2-8. validation + EarlyStopping + 학습곡선
# 이전(2-5)과 차이: #3에 validation_split + EarlyStopping 콜백 추가
#
# 왜 배우는가:
#   Phase 1-13 과적합을 코드로 감지하고 방지.
#   실무에서 model.fit()에 거의 항상 들어가는 것.
#
# 나중에 만나는 곳:
#   → 모든 이후 Phase의 #3: 콜백 필수
#
# ▶ 보고 오기: Coursera C2W3 "Advice for ML"
#
# Ref: AI5 keras15~19
# ============================================

#0. 라이브러리 ──────────────────────────────
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping       # ← NEW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing
import numpy as np
import matplotlib.pyplot as plt

#1. 데이터 ─────────────────────────────────
dataset = fetch_california_housing()
x = dataset.data
y = dataset.target

#2. 데이터 전처리 및 분할 ───────────────────
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#3. 모델 ──────────────────────────────────
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

# ── EarlyStopping 콜백 ──────────────────  ← NEW
es = EarlyStopping(
    monitor='val_loss',           # 검증 손실 감시
    patience=20,                  # 20 epoch 연속 개선 없으면 멈춤
    restore_best_weights=True,    # 가장 좋았던 가중치로 복원
    verbose=1,
)

history = model.fit(
    x_train, y_train,
    epochs=500,                   # 크게 잡아도 ES가 알아서 멈춤
    batch_size=32,
    validation_split=0.2,         # ← NEW: 20% 검증용
    callbacks=[es],               # ← NEW: 콜백 전달
    verbose=0,
)

# 실제 학습한 epoch 수
actual_epochs = len(history.history['loss'])
print(f"\n설정 epochs: 500, 실제 학습: {actual_epochs} epoch (EarlyStopping)")

#4. 평가 ──────────────────────────────────
loss = model.evaluate(x_test, y_test, verbose=0)
y_pred = model.predict(x_test, verbose=0).ravel()
r2 = r2_score(y_test, y_pred)

print(f"\n[ 평가 ]")
print(f"Test MSE: {loss:.4f}")
print(f"R²: {r2:.4f}")

# ── 학습곡선 시각화 ──────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Train vs Val Loss
axes[0].plot(history.history['loss'], 'b-', label='Train Loss')
axes[0].plot(history.history['val_loss'], 'r-', label='Val Loss')
axes[0].axvline(actual_epochs - 20, color='green', linestyle='--', alpha=0.5, label='Best Epoch')
axes[0].set_title('Learning Curve')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss (MSE)')
axes[0].legend()

# 과적합 감지 영역
train_loss = history.history['loss']
val_loss = history.history['val_loss']
gap = [v - t for t, v in zip(train_loss, val_loss)]
axes[1].plot(gap, 'orange', linewidth=2)
axes[1].axhline(0, color='gray', linestyle='--')
axes[1].fill_between(range(len(gap)), gap, 0, where=[g > 0 for g in gap], alpha=0.2, color='red')
axes[1].set_title('Val - Train Gap (Overfitting Signal)')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Gap')

# 예측 vs 실제
axes[2].scatter(y_test, y_pred, alpha=0.3, s=5)
axes[2].plot([0, 5], [0, 5], 'r--')
axes[2].set_title(f'Actual vs Predicted (R²={r2:.3f})')
axes[2].set_xlabel('Actual')
axes[2].set_ylabel('Predicted')

plt.tight_layout()
plt.savefig('2-8_output.png', dpi=100)
plt.show()

# ── 정리 ──────────────────────────────────
print("\n" + "="*50)
print("핵심 정리:")
print("  validation_split=0.2 → 검증 데이터 20% 분리")
print("  EarlyStopping → val_loss 감시, 자동 멈춤")
print("  restore_best_weights → 최적 가중치 복원")
print("  학습곡선 → train↓ val↑ = 과적합!")
print("  → 이후 모든 코드에 ES가 들어간다")
print("="*50)
print("\n★ Phase 2 완료! → 체크포인트 시험 2")
