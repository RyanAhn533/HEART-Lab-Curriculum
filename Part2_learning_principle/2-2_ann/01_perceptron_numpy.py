"""
========================================
01-01. Perceptron (퍼셉트론) - numpy 손코딩
========================================
ANN의 가장 기본 단위: 하나의 뉴런

구조: inputs → weighted sum → activation → output
수식: output = activation(Σ(w_i * x_i) + b)

이 파일에서 배우는 것:
1. 단일 퍼셉트론으로 AND, OR 게이트 학습
2. Forward pass 손계산
3. Weight update 손계산 (10 epoch)
4. XOR은 왜 안 되는가? → MLP의 필요성

★ 종이에 먼저 3 epoch 손계산 후 코드 실행 ★
"""

import numpy as np

# ==============================================
# 1단계: 단일 퍼셉트론 - AND 게이트
# ==============================================
print("=" * 50)
print("AND 게이트 학습")
print("=" * 50)

# AND 진리표
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype=float)

y_and = np.array([0, 0, 0, 1], dtype=float)  # AND

# 가중치 초기화
np.random.seed(42)
w = np.random.randn(2) * 0.1  # 가중치 2개 (입력이 2개니까)
b = 0.0
lr = 0.1

def sigmoid(z):
    """활성화 함수: 출력을 0~1 사이로 변환"""
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    """sigmoid의 미분 = a * (1 - a)"""
    return a * (1 - a)

print(f"초기 가중치: w = {w}, b = {b:.4f}")
print(f"\n{'Epoch':>5} | {'Loss':>8} | {'w1':>8} | {'w2':>8} | {'b':>8} | 예측")
print("-" * 70)

"""
★★★ 손계산 가이드 ★★★

Epoch 1, 데이터 [0,0] → y=0:
  1) z = w1*0 + w2*0 + b = b
  2) pred = sigmoid(z)
  3) error = pred - y
  4) Chain Rule:
     dL/dw1 = error * sigmoid'(z) * x1
     dL/dw2 = error * sigmoid'(z) * x2
     dL/db  = error * sigmoid'(z) * 1
  5) w1 = w1 - lr * dL/dw1
     w2 = w2 - lr * dL/dw2
     b  = b  - lr * dL/db
"""

for epoch in range(10):
    total_loss = 0
    preds = []

    for i in range(len(X)):
        # Forward
        z = np.dot(w, X[i]) + b
        pred = sigmoid(z)
        preds.append(pred)

        # Loss (Binary Cross Entropy 간소화 → MSE 사용)
        error = pred - y_and[i]
        total_loss += error ** 2

        # Backward (Chain Rule)
        d_pred = error                    # dL/dpred = 2*error (상수 생략)
        d_z = d_pred * sigmoid_derivative(pred)  # dpred/dz = sigmoid'
        dw = d_z * X[i]                   # dz/dw = x
        db_val = d_z                      # dz/db = 1

        # Update
        w = w - lr * dw
        b = b - lr * db_val

    avg_loss = total_loss / len(X)
    pred_str = " ".join([f"{p:.2f}" for p in preds])
    print(f"{epoch+1:5d} | {avg_loss:8.4f} | {w[0]:8.4f} | {w[1]:8.4f} | {b:8.4f} | [{pred_str}]")

print(f"\n학습 완료!")
print(f"AND 게이트 결과:")
for i in range(len(X)):
    z = np.dot(w, X[i]) + b
    pred = sigmoid(z)
    print(f"  {X[i]} → {pred:.4f} (정답: {y_and[i]:.0f}) {'✓' if round(pred) == y_and[i] else '✗'}")

# ==============================================
# 2단계: XOR - 퍼셉트론의 한계
# ==============================================
print("\n" + "=" * 50)
print("XOR 게이트 시도 (실패하는 것을 확인)")
print("=" * 50)

y_xor = np.array([0, 1, 1, 0], dtype=float)

w_xor = np.random.randn(2) * 0.1
b_xor = 0.0

for epoch in range(1000):  # 아무리 많이 해도 안 됨
    for i in range(len(X)):
        z = np.dot(w_xor, X[i]) + b_xor
        pred = sigmoid(z)
        error = pred - y_xor[i]
        d_z = error * sigmoid_derivative(pred)
        w_xor = w_xor - lr * d_z * X[i]
        b_xor = b_xor - lr * d_z

print("XOR 결과 (1000 epoch 후):")
for i in range(len(X)):
    z = np.dot(w_xor, X[i]) + b_xor
    pred = sigmoid(z)
    print(f"  {X[i]} → {pred:.4f} (정답: {y_xor[i]:.0f}) {'✓' if round(pred) == y_xor[i] else '✗'}")

print("\n→ 단일 퍼셉트론으로는 XOR을 풀 수 없습니다!")
print("→ 이유: XOR은 선형 분리가 불가능 (linearly non-separable)")
print("→ 해결: 층을 쌓자 → Multi-Layer Perceptron (MLP)")

# ==============================================
# ★ 과제 ★
# ==============================================
"""
[손계산 과제]
1. AND 게이트에서 w=[0.1, 0.1], b=0, lr=0.1로 시작하여
   첫 번째 epoch의 4개 데이터 각각에 대해 forward/backward를 종이에 계산하세요.

2. OR 게이트(y=[0,1,1,1])로 바꿔서 학습시켜 보세요.

3. XOR이 왜 안 되는지 2차원 평면에 점을 찍어서 직선 하나로
   분리할 수 없음을 그림으로 설명하세요.

[코딩 과제]
4. NAND 게이트(y=[1,1,1,0])를 학습시켜 보세요.
5. sigmoid 대신 step function(z>0이면 1, 아니면 0)을 쓰면 어떻게 되나요?
"""
