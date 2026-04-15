# ============================================
# 1-8. 손실함수 — MSE (Mean Squared Error)
#
# 왜 배우는가:
#   "모델이 얼마나 틀렸는가"를 숫자 하나로 표현.
#   이 숫자를 줄이는 것이 학습(training).
#
# 나중에 만나는 곳:
#   → Phase 5: model.compile(loss='mse')
#   → 1-9: 경사하강법이 이 값을 줄인다
#   → 1-11: 분류용 손실함수 (BCE, CCE)
#
# ▶ 보고 오기: Coursera C1W1 "Cost Function"
#
# Ref: Stanford CS229 W2 / Google MLCC "Loss"
# ============================================

import numpy as np
import matplotlib.pyplot as plt

# ── 1. MSE 직접 계산 ─────────────────────
print("[ MSE 직접 계산 ]")

actual =    np.array([3.0, 2.5, 4.0, 3.5, 5.0])
predicted = np.array([2.8, 2.7, 3.5, 3.6, 4.8])
errors = actual - predicted

print(f"실제값:  {actual}")
print(f"예측값:  {predicted}")
print(f"오차:    {errors}")
print(f"오차²:   {errors**2}")

mse = np.mean(errors**2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(errors))

print(f"\nMSE  = {mse:.4f}")
print(f"RMSE = {rmse:.4f}  ← 원래 단위 (해석 쉬움)")
print(f"MAE  = {mae:.4f}")
print(f"\n→ 평균적으로 {rmse:.2f}만큼 틀린다")

# ── 2. w를 바꾸면 MSE가 어떻게 변하나 ────
print(f"\n[ w를 바꿀 때 MSE 변화 ]")

# 간단한 데이터: y = 2x (정답 w=2)
x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 6, 8, 10], dtype=float)

w_values = np.linspace(0, 4, 50)
mse_values = []

for w in w_values:
    y_pred = w * x
    mse_val = np.mean((y - y_pred)**2)
    mse_values.append(mse_val)

best_w = w_values[np.argmin(mse_values)]
print(f"최적 w = {best_w:.2f} (정답: 2.0)")
print(f"최소 MSE = {min(mse_values):.4f}")

# ── 3. 2D 손실 곡면 (w와 b 동시) ────────
w_range = np.linspace(0, 4, 50)
b_range = np.linspace(-3, 3, 50)
W, B = np.meshgrid(w_range, b_range)
MSE_surface = np.zeros_like(W)

for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        y_pred = W[i, j] * x + B[i, j]
        MSE_surface[i, j] = np.mean((y - y_pred)**2)

# ── 4. 시각화 ─────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# 오차 시각화
axes[0, 0].scatter(range(len(actual)), actual, color='blue', s=80, label='Actual', zorder=5)
axes[0, 0].scatter(range(len(predicted)), predicted, color='red', s=80, label='Predicted', zorder=5)
for i in range(len(actual)):
    axes[0, 0].plot([i, i], [actual[i], predicted[i]], 'g--', linewidth=2)
axes[0, 0].set_title('Actual vs Predicted (errors in green)')
axes[0, 0].legend()

# 오차 크기
axes[0, 1].bar(range(len(errors)), errors**2, color='orange', edgecolor='black')
axes[0, 1].axhline(mse, color='red', linestyle='--', label=f'MSE={mse:.3f}')
axes[0, 1].set_title('Squared Errors')
axes[0, 1].set_xlabel('Data Point')
axes[0, 1].set_ylabel('Error²')
axes[0, 1].legend()

# w vs MSE (1D 손실 곡선)
axes[0, 2].plot(w_values, mse_values, 'b-', linewidth=2)
axes[0, 2].scatter([best_w], [min(mse_values)], color='red', s=100, zorder=5, label=f'Best w={best_w:.2f}')
axes[0, 2].set_xlabel('w (weight)')
axes[0, 2].set_ylabel('MSE')
axes[0, 2].set_title('Loss Curve: w vs MSE')
axes[0, 2].legend()

# 2D 손실 곡면 (contour)
cs = axes[1, 0].contourf(W, B, MSE_surface, levels=20, cmap='viridis')
axes[1, 0].scatter([2], [0], color='red', s=100, zorder=5, label='Optimal (w=2, b=0)')
axes[1, 0].set_xlabel('w')
axes[1, 0].set_ylabel('b')
axes[1, 0].set_title('Loss Surface (contour)')
axes[1, 0].legend()
plt.colorbar(cs, ax=axes[1, 0])

# MSE vs RMSE vs MAE 비교
metrics = {'MSE': mse, 'RMSE': rmse, 'MAE': mae}
axes[1, 1].bar(metrics.keys(), metrics.values(), color=['orange', 'green', 'blue'], edgecolor='black')
axes[1, 1].set_title('MSE vs RMSE vs MAE')

# 큰 오차의 영향
errors_small = np.array([0.1, 0.2, 0.1, 0.2, 0.1])
errors_big = np.array([0.1, 0.2, 0.1, 0.2, 2.0])  # 하나만 큼
mse_small = np.mean(errors_small**2)
mse_big = np.mean(errors_big**2)
axes[1, 2].bar(['All Small Errors', 'One Big Error'], [mse_small, mse_big],
               color=['green', 'red'], edgecolor='black')
axes[1, 2].set_title(f'MSE: Small={mse_small:.3f} vs Big={mse_big:.3f}')
axes[1, 2].set_ylabel('MSE')

plt.tight_layout()
plt.savefig('1-8_output.png', dpi=100)
plt.show()

# ── 정리 ──────────────────────────────────
print("\n" + "="*50)
print("핵심 정리:")
print("  MSE = 오차²의 평균 (큰 오차에 페널티)")
print("  RMSE = √MSE (원래 단위로 해석)")
print("  MAE = |오차|의 평균 (이상치에 덜 민감)")
print("  w를 바꾸면 MSE가 변한다 → 최소점 찾기 = 학습")
print("  → 이 최소점을 찾는 방법 = 경사하강법 (1-9)")
print("="*50)
