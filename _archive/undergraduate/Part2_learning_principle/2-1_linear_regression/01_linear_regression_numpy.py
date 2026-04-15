"""
========================================
00-01. 인공지능이란 무엇인가?
========================================
정의: Prediction(예측값)과 Ground Truth(실제값)의 오차를 줄여나가는 과정

이 파일에서 배우는 것:
1. 가장 단순한 모델: y = wx + b
2. Loss(오차) 계산: MSE = (pred - y)^2 / n
3. Gradient Descent: w를 어느 방향으로 얼마나 움직일지
4. 10 epoch 손계산 → 코드로 검증

★ 먼저 종이에 직접 계산한 뒤 코드를 실행하세요 ★
"""

import numpy as np

# ==============================================
# 1단계: 데이터 준비 (아주 간단한 예시)
# ==============================================
# x: 공부시간, y: 시험점수
x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([3, 5, 7, 9, 11], dtype=float)  # 정답: y = 2x + 1

print("=" * 50)
print("데이터 확인")
print("=" * 50)
for i in range(len(x)):
    print(f"  공부 {x[i]}시간 → 점수 {y[i]}")

# ==============================================
# 2단계: 모델 초기화
# ==============================================
# y = wx + b 에서 w와 b를 아무 값으로 시작
w = 0.0  # 가중치 (weight)
b = 0.0  # 편향 (bias)
lr = 0.01  # 학습률 (learning rate)
epochs = 10

print(f"\n초기값: w = {w}, b = {b}, lr = {lr}")

# ==============================================
# 3단계: 10 epoch 학습 (손계산과 동일한 과정)
# ==============================================
"""
★★★ 손계산 가이드 (종이에 먼저 해보세요) ★★★

[Forward Pass]
  pred = w * x + b
  각 데이터에 대해 pred 계산

[Loss 계산]
  MSE = (1/n) * Σ(pred_i - y_i)^2

[Backward Pass - Chain Rule]
  dL/dw = (2/n) * Σ(pred_i - y_i) * x_i
  dL/db = (2/n) * Σ(pred_i - y_i)

  왜? Chain Rule:
  L = (pred - y)^2
  dL/dpred = 2(pred - y)
  dpred/dw = x
  따라서 dL/dw = dL/dpred * dpred/dw = 2(pred - y) * x

[Weight Update]
  w = w - lr * dL/dw
  b = b - lr * dL/db
"""

print("\n" + "=" * 50)
print("학습 시작 (10 epochs)")
print("=" * 50)

for epoch in range(epochs):
    # --- Forward ---
    pred = w * x + b

    # --- Loss (MSE) ---
    loss = np.mean((pred - y) ** 2)

    # --- Backward (Gradient 계산) ---
    # Chain Rule 적용
    dL_dpred = 2 * (pred - y) / len(x)  # Loss를 pred로 미분
    dw = np.sum(dL_dpred * x)            # pred를 w로 미분하면 x
    db = np.sum(dL_dpred)                # pred를 b로 미분하면 1

    # --- Update ---
    w = w - lr * dw
    b = b - lr * db

    print(f"  Epoch {epoch+1:2d} | Loss: {loss:8.4f} | w: {w:.4f} | b: {b:.4f} | pred[0]: {pred[0]:.4f}")

print(f"\n최종 결과: y = {w:.4f}x + {b:.4f}")
print(f"정답:      y = 2.0000x + 1.0000")

# ==============================================
# 4단계: 학습 결과 시각화
# ==============================================
try:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))

    # 왼쪽: 데이터와 학습된 직선
    plt.subplot(1, 2, 1)
    plt.scatter(x, y, color='red', s=100, zorder=5, label='실제값 (Ground Truth)')
    x_line = np.linspace(0, 6, 100)
    plt.plot(x_line, w * x_line + b, 'b-', linewidth=2, label=f'예측: y={w:.2f}x+{b:.2f}')
    plt.xlabel('x (공부시간)')
    plt.ylabel('y (점수)')
    plt.title('Linear Regression 결과')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 오른쪽: Loss 변화 (다시 학습하면서 기록)
    plt.subplot(1, 2, 2)
    w_temp, b_temp = 0.0, 0.0
    losses = []
    for _ in range(100):
        pred_temp = w_temp * x + b_temp
        loss_temp = np.mean((pred_temp - y) ** 2)
        losses.append(loss_temp)
        dw_temp = np.mean(2 * (pred_temp - y) * x)
        db_temp = np.mean(2 * (pred_temp - y))
        w_temp -= lr * dw_temp
        b_temp -= lr * db_temp

    plt.plot(losses, 'g-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Loss가 줄어드는 과정 = AI가 학습하는 과정')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('C:/Users/Ryan/AI5/curriculum/00_what_is_ai/loss_curve.png', dpi=100)
    plt.show()
    print("\n그래프가 저장되었습니다.")
except ImportError:
    print("\nmatplotlib이 없어 시각화를 건너뜁니다.")

# ==============================================
# ★ 과제 ★
# ==============================================
"""
[손계산 과제]
1. w=0, b=0, lr=0.01 에서 시작하여 3 epoch를 종이에 직접 계산하세요.
   - 각 epoch마다: pred 5개, loss, dw, db, 업데이트된 w와 b

2. lr=0.01 대신 lr=0.1로 바꾸면 어떻게 되는가? 종이에 계산 후 코드로 확인.

3. lr=1.0으로 하면? → 발산(diverge)하는 이유를 Chain Rule로 설명하세요.

[코딩 과제]
4. 데이터를 y = 3x + 2 로 바꾸고 학습시켜 보세요.
5. 데이터에 노이즈를 추가하면 (y + random) 결과가 어떻게 달라지나요?
"""
