# ============================================
# 1-9. 경사하강법 (Gradient Descent)
#
# 왜 배우는가:
#   손실(MSE)을 줄이는 방법.
#   모든 딥러닝 학습의 핵심 원리.
#
# 나중에 만나는 곳:
#   → Phase 5: optimizer='adam'
#   → Phase 10: ReduceLROnPlateau
#
# ▶ 보고 오기: 3B1B "Gradient descent" (필수!)
#
# Ref: Stanford CS229 W2 / Coursera C1W1
# ============================================

import numpy as np
import matplotlib.pyplot as plt

# ── 1. 데이터 (y = 2x, 정답 w=2) ────────
x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 6, 8, 10], dtype=float)

# ── 2. 경사하강법 직접 구현 ──────────────
def gradient_descent(x, y, lr, epochs):
    w = 0.0  # 초기 가중치
    b = 0.0  # 초기 편향
    n = len(x)
    history = {'w': [], 'b': [], 'mse': []}

    for epoch in range(epochs):
        # 순전파: 예측
        y_pred = w * x + b

        # 손실 계산
        mse = np.mean((y - y_pred)**2)

        # 기울기(gradient) 계산
        dw = -(2/n) * np.sum(x * (y - y_pred))
        db = -(2/n) * np.sum(y - y_pred)

        # 가중치 업데이트
        w = w - lr * dw
        b = b - lr * db

        history['w'].append(w)
        history['b'].append(b)
        history['mse'].append(mse)

        if epoch % 20 == 0 or epoch == epochs-1:
            print(f"  Epoch {epoch:3d}: w={w:.4f}, b={b:.4f}, MSE={mse:.4f}")

    return w, b, history

# ── 3. 적당한 lr ─────────────────────────
print("[ lr=0.01 — 적당한 학습률 ]")
w_good, b_good, hist_good = gradient_descent(x, y, lr=0.01, epochs=200)

# ── 4. 너무 큰 lr ────────────────────────
print(f"\n[ lr=0.1 — 너무 큰 학습률 ]")
w_big, b_big, hist_big = gradient_descent(x, y, lr=0.1, epochs=20)

# ── 5. 너무 작은 lr ──────────────────────
print(f"\n[ lr=0.001 — 너무 작은 학습률 ]")
w_small, b_small, hist_small = gradient_descent(x, y, lr=0.001, epochs=200)

# ── 6. 시각화 ─────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# 학습 곡선 (MSE vs Epoch) — 적당한 lr
axes[0, 0].plot(hist_good['mse'], 'b-', linewidth=2)
axes[0, 0].set_title('lr=0.01 (Good)')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('MSE')
axes[0, 0].set_ylim(bottom=0)

# 학습 곡선 — 큰 lr
axes[0, 1].plot(hist_big['mse'], 'r-', linewidth=2)
axes[0, 1].set_title('lr=0.1 (Too Big)')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('MSE')

# 학습 곡선 — 작은 lr
axes[0, 2].plot(hist_small['mse'], 'g-', linewidth=2)
axes[0, 2].set_title('lr=0.001 (Too Small)')
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('MSE')
axes[0, 2].set_ylim(bottom=0)

# w의 변화 과정
axes[1, 0].plot(hist_good['w'], 'b-', label='lr=0.01')
axes[1, 0].plot(hist_small['w'], 'g-', label='lr=0.001')
axes[1, 0].axhline(2.0, color='red', linestyle='--', label='Optimal w=2')
axes[1, 0].set_title('w Convergence')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('w')
axes[1, 0].legend()

# 손실 곡면 위에서 경사하강법 경로
w_range = np.linspace(-0.5, 4, 100)
mse_curve = [np.mean((y - w_val * x)**2) for w_val in w_range]
axes[1, 1].plot(w_range, mse_curve, 'k-', linewidth=2)
axes[1, 1].plot(hist_good['w'], hist_good['mse'], 'bo-', markersize=3, alpha=0.5, label='GD path')
axes[1, 1].scatter([2], [0], color='red', s=100, zorder=5, label='Optimal')
axes[1, 1].set_title('GD Path on Loss Curve')
axes[1, 1].set_xlabel('w')
axes[1, 1].set_ylabel('MSE')
axes[1, 1].legend()

# 최종 결과 비교
axes[1, 2].scatter(x, y, color='blue', s=80, label='Data', zorder=5)
x_line = np.linspace(0, 6, 50)
axes[1, 2].plot(x_line, w_good * x_line + b_good, 'b-', linewidth=2,
                label=f'GD: y={w_good:.2f}x+{b_good:.2f}')
axes[1, 2].plot(x_line, 2 * x_line, 'r--', linewidth=1, label='y=2x (answer)')
axes[1, 2].set_title('Final Result')
axes[1, 2].set_xlabel('x')
axes[1, 2].set_ylabel('y')
axes[1, 2].legend()

plt.tight_layout()
plt.savefig('1-9_output.png', dpi=100)
plt.show()

# ── 정리 ──────────────────────────────────
print("\n" + "="*50)
print("핵심 정리:")
print(f"  경사하강법 = 기울기 방향으로 한 발짝씩 이동")
print(f"  w_new = w_old - lr * gradient")
print(f"  lr 너무 크면 → 발산, 너무 작으면 → 느림")
print(f"  최종 결과: w={w_good:.4f} (정답: 2.0)")
print(f"  → 이게 optimizer='adam'이 하는 일")
print("="*50)
