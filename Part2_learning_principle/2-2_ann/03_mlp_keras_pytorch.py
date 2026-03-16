"""
========================================
01-03. MLP를 프레임워크로 구현 (Keras & PyTorch)
========================================
앞에서 numpy로 손코딩한 것과 동일한 MLP를
Keras와 PyTorch로 각각 구현하여 비교합니다.

느낄 점: "아 이게 프레임워크가 자동으로 해주는 거구나"
- Forward: model(x) 한 줄
- Backward: loss.backward() 한 줄 (우리가 종이에 계산한 그것)
- Update: optimizer.step() 한 줄
"""

import numpy as np

# ==============================================
# 1. Keras로 XOR 풀기
# ==============================================
print("=" * 50)
print("Keras로 XOR")
print("=" * 50)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[0],[1],[1],[0]], dtype=float)

model_keras = Sequential([
    Dense(4, activation='sigmoid', input_shape=(2,)),  # Hidden layer
    Dense(1, activation='sigmoid')                      # Output layer
])
# 우리가 손으로 한 것: MSE loss + gradient descent
model_keras.compile(optimizer='sgd', loss='mse')

print("학습 시작...")
history = model_keras.fit(X, y, epochs=10000, verbose=0)

print(f"최종 Loss: {history.history['loss'][-1]:.6f}")
pred = model_keras.predict(X, verbose=0)
for i in range(4):
    print(f"  {X[i]} → {pred[i,0]:.4f} (정답: {y[i,0]:.0f})")

# ==============================================
# 2. PyTorch로 XOR 풀기
# ==============================================
print("\n" + "=" * 50)
print("PyTorch로 XOR")
print("=" * 50)

import torch
import torch.nn as nn

X_t = torch.FloatTensor(X)
y_t = torch.FloatTensor(y)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 4)   # 우리가 W1, b1이라 부른 것
        self.layer2 = nn.Linear(4, 1)   # 우리가 W2, b2라 부른 것
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.layer1(x))  # z1=XW1+b1, a1=sigmoid(z1)
        x = self.sigmoid(self.layer2(x))  # z2=a1W2+b2, a2=sigmoid(z2)
        return x

model_torch = MLP()
criterion = nn.MSELoss()                        # 우리가 직접 계산한 MSE
optimizer = torch.optim.SGD(model_torch.parameters(), lr=0.5)  # Gradient Descent

for epoch in range(10000):
    # Forward (우리가 z1, a1, z2, a2 계산한 것)
    pred = model_torch(X_t)

    # Loss (우리가 (pred-y)^2 계산한 것)
    loss = criterion(pred, y_t)

    # Backward (우리가 delta2, delta1 Chain Rule 전개한 것)
    optimizer.zero_grad()
    loss.backward()    # ← 이 한 줄이 종이 2장 분량의 Chain Rule

    # Update (우리가 W -= lr * dW 한 것)
    optimizer.step()

print(f"최종 Loss: {loss.item():.6f}")
pred = model_torch(X_t).detach().numpy()
for i in range(4):
    print(f"  {X[i]} → {pred[i,0]:.4f} (정답: {y[i,0]:.0f})")

# ==============================================
# 3. 비교 정리
# ==============================================
print("\n" + "=" * 50)
print("numpy vs Keras vs PyTorch 비교")
print("=" * 50)
print("""
┌──────────────┬────────────────────┬──────────────────┬──────────────────┐
│   단계       │ numpy (손코딩)      │ Keras            │ PyTorch          │
├──────────────┼────────────────────┼──────────────────┼──────────────────┤
│ Forward      │ z1 = X @ W1 + b1   │ model(X)         │ model(X_t)       │
│              │ a1 = sigmoid(z1)    │ (자동)            │ (자동)            │
│              │ z2 = a1 @ W2 + b2   │                  │                  │
│              │ a2 = sigmoid(z2)    │                  │                  │
├──────────────┼────────────────────┼──────────────────┼──────────────────┤
│ Loss         │ mean((a2-y)^2)      │ compile(loss=)   │ criterion(p, y)  │
├──────────────┼────────────────────┼──────────────────┼──────────────────┤
│ Backward     │ delta2 = ...        │ (자동)            │ loss.backward()  │
│ (Chain Rule) │ delta1 = ...        │                  │                  │
│              │ dW2 = a1.T @ d2     │                  │                  │
│              │ dW1 = X.T @ d1      │                  │                  │
├──────────────┼────────────────────┼──────────────────┼──────────────────┤
│ Update       │ W -= lr * dW        │ (자동)            │ optimizer.step() │
└──────────────┴────────────────────┴──────────────────┴──────────────────┘

핵심: 프레임워크가 해주는 건 결국 우리가 손으로 한 것과 "정확히 같은 수학"입니다.
""")

# ==============================================
# ★ 과제 ★
# ==============================================
"""
1. Keras와 PyTorch 각각에서 model.summary() / print(model)로
   파라미터 수를 확인하세요. 우리가 직접 만든 W1(2x4), b1(4), W2(4x1), b2(1)과 일치하나요?

2. optimizer를 SGD → Adam으로 바꾸면 수렴 속도가 어떻게 달라지나요?

3. hidden_size를 4 → 8 → 16으로 늘리면 XOR 수렴이 빨라지나요?
   파라미터 수는 각각 몇 개인지 손으로 계산하세요.
"""
