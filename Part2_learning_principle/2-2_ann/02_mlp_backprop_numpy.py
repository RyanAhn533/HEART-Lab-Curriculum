"""
========================================
01-02. MLP 역전파 (Backpropagation) - numpy 손코딩
========================================
XOR을 풀기 위해 층을 쌓는다 → Multi-Layer Perceptron

구조: Input(2) → Hidden(4, sigmoid) → Output(1, sigmoid)

이 파일에서 배우는 것:
1. 다층 신경망의 Forward Pass
2. Chain Rule을 이용한 역전파 (Backpropagation) 전체 과정
3. 10 epoch 손계산 검증
4. XOR 문제 해결

★ 반드시 Chain Rule 전개를 종이에 먼저 써보세요 ★
"""

import numpy as np

np.random.seed(42)

# ==============================================
# 1단계: XOR 데이터
# ==============================================
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype=float)

y = np.array([[0], [1], [1], [0]], dtype=float)

# ==============================================
# 2단계: 네트워크 구조
# ==============================================
# Input(2) → Hidden(4) → Output(1)
input_size = 2
hidden_size = 4
output_size = 1
lr = 0.5

# 가중치 초기화
W1 = np.random.randn(input_size, hidden_size) * 0.5   # (2, 4)
b1 = np.zeros((1, hidden_size))                        # (1, 4)
W2 = np.random.randn(hidden_size, output_size) * 0.5   # (4, 1)
b2 = np.zeros((1, output_size))                        # (1, 1)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(a):
    return a * (1 - a)

# ==============================================
# 3단계: 학습 (10 epoch 상세 출력)
# ==============================================
"""
★★★ 역전파 Chain Rule 전개 ★★★

네트워크: X → z1=XW1+b1 → a1=sigmoid(z1) → z2=a1W2+b2 → a2=sigmoid(z2) → Loss

Loss = (1/2)(a2 - y)^2

[Output Layer]
  dL/da2 = (a2 - y)
  da2/dz2 = sigmoid'(a2) = a2(1-a2)
  → delta2 = dL/dz2 = (a2 - y) * sigmoid'(a2)

  dz2/dW2 = a1
  → dL/dW2 = a1^T · delta2

  dz2/db2 = 1
  → dL/db2 = sum(delta2)

[Hidden Layer] ← Chain Rule 계속 전파
  dz2/da1 = W2
  → dL/da1 = delta2 · W2^T

  da1/dz1 = sigmoid'(a1)
  → delta1 = dL/da1 * sigmoid'(a1) = (delta2 · W2^T) * a1(1-a1)

  dz1/dW1 = X
  → dL/dW1 = X^T · delta1

  dz1/db1 = 1
  → dL/db1 = sum(delta1)
"""

print("=" * 60)
print("MLP 역전파 학습 (XOR)")
print("구조: Input(2) → Hidden(4, sigmoid) → Output(1, sigmoid)")
print("=" * 60)

losses = []

for epoch in range(10):
    # ===== Forward Pass =====
    z1 = X @ W1 + b1          # (4, 4) = (4,2) @ (2,4)
    a1 = sigmoid(z1)           # (4, 4)

    z2 = a1 @ W2 + b2         # (4, 1) = (4,4) @ (4,1)
    a2 = sigmoid(z2)           # (4, 1) ← 최종 예측

    # ===== Loss =====
    loss = np.mean((a2 - y) ** 2)
    losses.append(loss)

    # ===== Backward Pass (Chain Rule) =====
    # Output layer
    delta2 = (a2 - y) * sigmoid_deriv(a2)  # (4, 1)
    dW2 = a1.T @ delta2                     # (4, 1)
    db2 = np.sum(delta2, axis=0, keepdims=True)

    # Hidden layer (Chain Rule 전파)
    delta1 = (delta2 @ W2.T) * sigmoid_deriv(a1)  # (4, 4)
    dW1 = X.T @ delta1                             # (2, 4)
    db1 = np.sum(delta1, axis=0, keepdims=True)

    # ===== Update =====
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    # 출력
    pred_str = " ".join([f"{a2[i,0]:.3f}" for i in range(4)])
    print(f"Epoch {epoch+1:2d} | Loss: {loss:.6f} | Pred: [{pred_str}] | 정답: [0 1 1 0]")

# ==============================================
# 4단계: 더 많은 epoch 학습
# ==============================================
print(f"\n--- 추가 학습 (총 10000 epoch) ---")
for epoch in range(10, 10000):
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)

    loss = np.mean((a2 - y) ** 2)
    losses.append(loss)

    delta2 = (a2 - y) * sigmoid_deriv(a2)
    dW2 = a1.T @ delta2
    db2 = np.sum(delta2, axis=0, keepdims=True)
    delta1 = (delta2 @ W2.T) * sigmoid_deriv(a1)
    dW1 = X.T @ delta1
    db1 = np.sum(delta1, axis=0, keepdims=True)

    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    if (epoch + 1) % 2000 == 0:
        pred_str = " ".join([f"{a2[i,0]:.3f}" for i in range(4)])
        print(f"Epoch {epoch+1:5d} | Loss: {loss:.6f} | Pred: [{pred_str}]")

# 최종 결과
print(f"\n최종 XOR 결과:")
z1 = X @ W1 + b1
a1 = sigmoid(z1)
z2 = a1 @ W2 + b2
a2 = sigmoid(z2)
for i in range(4):
    print(f"  {X[i]} → {a2[i,0]:.4f} → 반올림: {round(a2[i,0])} (정답: {int(y[i,0])}) {'✓' if round(a2[i,0]) == y[i,0] else '✗'}")

# ==============================================
# 5단계: 학습된 가중치 확인
# ==============================================
print(f"\n학습된 가중치:")
print(f"  W1 (2x4):\n{W1}")
print(f"  b1 (1x4): {b1}")
print(f"  W2 (4x1):\n{W2}")
print(f"  b2 (1x1): {b2}")

# ==============================================
# 시각화
# ==============================================
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss curve
    axes[0].plot(losses)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('XOR 학습 - Loss 감소 과정')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)

    # Decision boundary
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    z1_grid = grid @ W1 + b1
    a1_grid = sigmoid(z1_grid)
    z2_grid = a1_grid @ W2 + b2
    a2_grid = sigmoid(z2_grid).reshape(xx.shape)

    axes[1].contourf(xx, yy, a2_grid, levels=50, cmap='RdYlBu_r', alpha=0.8)
    axes[1].scatter(X[:, 0], X[:, 1], c=y.ravel(), s=200, edgecolors='black',
                    cmap='RdYlBu_r', linewidth=2, zorder=5)
    axes[1].set_title('XOR Decision Boundary (MLP가 비선형 경계를 학습)')
    axes[1].set_xlabel('x1')
    axes[1].set_ylabel('x2')

    plt.tight_layout()
    plt.savefig('C:/Users/Ryan/AI5/curriculum/01_ANN/mlp_xor_result.png', dpi=100)
    plt.show()
except ImportError:
    pass

# ==============================================
# ★ 과제 ★
# ==============================================
"""
[손계산 과제 - 가장 중요!!!]
1. W1=[[0.5, -0.5], [0.3, -0.3]], b1=[0, 0], W2=[[0.5], [-0.5]], b2=[0]
   입력 [1, 0] (정답 1)에 대해:

   (a) Forward: z1, a1, z2, a2 각각 계산
   (b) Loss 계산
   (c) Backward: delta2, dW2, delta1, dW1 각각 Chain Rule로 유도하고 계산
   (d) Update: lr=0.5로 새로운 W1, W2, b1, b2 계산

   이걸 3 epoch 반복하세요.

2. Hidden layer의 뉴런 수를 2개로 줄이면 XOR을 풀 수 있나요?
   왜 / 왜 안 되는지 설명하세요.

[코딩 과제]
3. ReLU (max(0, z))로 activation을 바꿔보세요. 미분은 z>0이면 1, 아니면 0.
4. Hidden layer를 2개 (Input→Hidden1→Hidden2→Output)로 늘려보세요.
   Chain Rule이 어떻게 확장되는지 확인하세요.
"""
