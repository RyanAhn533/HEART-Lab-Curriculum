# ============================================
# B-5. 캡스톤 — 손 역전파 검증 + XOR 실험
# 이전(Chapter 3)과 차이: 라이브러리가 아니라 numpy로 신경망을 직접 굴린다
#
# 왜 배우는가:
#   model.fit()이 하는 일을 손으로 해 보면 더 이상 마법이 아니다.
#   XOR로 "비선형이 없으면 신경망이 아니다"를 눈으로 증명한다.
#
# ▶ 보고 오기: 3Blue1Brown "Backpropagation calculus" (Ch.4)
# ============================================

#0. 라이브러리 ──────────────────────────────
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

#1. 손계산 검증 — B-5 문제 1과 같은 숫자 ─────
print("=" * 44)
print("PART 1. 손계산 검증 (B-5 문제 1)")
print("=" * 44)

x  = np.array([1.0, 0.0]); t = 1.0
W1 = np.array([[0.5, 0.5], [0.5, 0.5]]); b1 = np.zeros(2)
W2 = np.array([0.5, 0.5]);               b2 = 0.0

# 순전파
z1 = W1 @ x + b1          # [0.5, 0.5]
a1 = sigmoid(z1)          # [0.6225, 0.6225]
y  = W2 @ a1 + b2         # 0.6225
L  = (y - t) ** 2         # 0.1425

print(f"z1 = {z1}")
print(f"a1 = {np.round(a1, 4)}")
print(f"y  = {y:.4f}")
print(f"L  = {L:.4f}")

# 역전파 (연쇄법칙 — 계산그래프를 거꾸로)
dL_dy  = 2 * (y - t)                  # dL/dy
dL_dW2 = dL_dy * a1                   # dL/dW2 = dL/dy · dy/dW2
dL_da1 = dL_dy * W2                   # 뒤로 전달
dL_dz1 = dL_da1 * a1 * (1 - a1)       # σ' = σ(1−σ)
dL_dW1 = np.outer(dL_dz1, x)          # dL/dW1

print(f"dL/dW2 = {np.round(dL_dW2, 4)}")
print(f"dL/dW1 =\n{np.round(dL_dW1, 4)}")
print("→ 네 손계산과 이 숫자가 일치해야 통과")

#2. 데이터 — XOR ────────────────────────────
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
T = np.array([0., 1., 1., 0.])

#3. 모델 (a) 선형 모델 — 실패를 확인 ─────────
print()
print("=" * 44)
print("PART 2(a). 활성화 없는 선형 모델로 XOR")
print("=" * 44)

rng = np.random.default_rng(0)
w = rng.normal(0, 0.5, 2); b = 0.0
for epoch in range(3000):
    y = X @ w + b                     # 활성화 없음 = 그냥 직선
    grad_y = 2 * (y - T) / len(T)
    w -= 0.1 * (X.T @ grad_y)
    b -= 0.1 * grad_y.sum()
pred = X @ w + b
print("예측:", np.round(pred, 3), "→ 정답:", T)
print("→ 아무리 돌려도 전부 0.5 근처. 직선 하나로는 XOR을 못 가른다")

#3. 모델 (b) 은닉층 + sigmoid — 성공을 확인 ──
print()
print("=" * 44)
print("PART 2(b). 은닉 2 + sigmoid 넣고 XOR")
print("=" * 44)

W1 = rng.normal(0, 1.0, (2, 2)); b1 = np.zeros(2)
W2 = rng.normal(0, 1.0, 2);      b2 = 0.0
lr = 0.5
for epoch in range(20000):
    # 순전파 (PART 1과 똑같은 계산, 데이터 4개 동시에)
    Z1 = X @ W1.T + b1
    A1 = sigmoid(Z1)
    Y  = A1 @ W2 + b2
    # 역전파 (PART 1과 똑같은 연쇄법칙)
    dY  = 2 * (Y - T) / len(T)
    dW2 = A1.T @ dY
    dA1 = np.outer(dY, W2)
    dZ1 = dA1 * A1 * (1 - A1)
    dW1 = dZ1.T @ X
    # 경사하강
    W2 -= lr * dW2; b2 -= lr * dY.sum()
    W1 -= lr * dW1; b1 -= lr * dZ1.sum(axis=0)

Y = sigmoid(X @ W1.T + b1) @ W2 + b2
print("예측:", np.round(Y, 3), "→ 정답:", T)

#4. 평가 ──────────────────────────────────
ok = np.all((Y > 0.5) == (T > 0.5))
print()
print("XOR 분류 성공!" if ok else "실패 — lr/epoch 바꿔서 재시도")
print("→ 비선형 한 스푼이 '층 쌓기'를 의미 있게 만들었다 (B-3 붕괴 증명의 역)")
