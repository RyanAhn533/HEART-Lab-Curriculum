"""
========================================
03-01. RNN (Recurrent Neural Network) - numpy 손코딩
========================================
ANN/CNN의 한계: 순서(시간) 정보를 다룰 수 없다
→ "나는 오늘 기분이 ___" 에서 다음 단어를 예측하려면 순서가 필요

RNN의 핵심: Hidden State (기억)
  h_t = tanh(W_xh * x_t + W_hh * h_(t-1) + b_h)
  y_t = W_hy * h_t + b_y

이 파일에서 배우는 것:
1. Hidden state가 어떻게 전달되는지 손계산
2. 3 timestep에 대한 Forward 전체 계산
3. BPTT (Backpropagation Through Time) 개념
4. Vanishing Gradient 문제 확인

★ h0=[0,0]에서 시작하여 3 timestep forward를 종이에 계산 ★
"""

import numpy as np

np.random.seed(42)

# ==============================================
# 1단계: 가장 간단한 RNN
# ==============================================
print("=" * 60)
print("RNN Forward Pass 손계산")
print("=" * 60)

# 하이퍼파라미터
input_size = 2    # 입력 차원
hidden_size = 3   # hidden state 차원
output_size = 1   # 출력 차원

# 가중치 (작은 값으로 초기화)
W_xh = np.random.randn(input_size, hidden_size) * 0.1   # (2, 3) 입력→hidden
W_hh = np.random.randn(hidden_size, hidden_size) * 0.1  # (3, 3) hidden→hidden
b_h = np.zeros(hidden_size)                               # (3,)

W_hy = np.random.randn(hidden_size, output_size) * 0.1  # (3, 1) hidden→출력
b_y = np.zeros(output_size)                               # (1,)

# 입력 시퀀스: 3개의 timestep, 각각 2차원
X_seq = np.array([
    [1.0, 0.5],   # t=0
    [0.3, 0.8],   # t=1
    [0.7, 0.2],   # t=2
])

print(f"입력 시퀀스 (3 timesteps, 2 features):")
for t in range(3):
    print(f"  t={t}: {X_seq[t]}")

print(f"\n가중치:")
print(f"  W_xh (input→hidden): shape {W_xh.shape}")
print(f"  W_hh (hidden→hidden): shape {W_hh.shape}")
print(f"  W_hy (hidden→output): shape {W_hy.shape}")

# ==============================================
# 2단계: Forward Pass (상세)
# ==============================================
"""
★★★ 손계산 가이드 ★★★

[t=0] h_prev = [0, 0, 0] (초기 hidden state)
  z_h = x_0 @ W_xh + h_prev @ W_hh + b_h
      = [1.0, 0.5] @ W_xh + [0,0,0] @ W_hh + [0,0,0]
  h_0 = tanh(z_h)
  y_0 = h_0 @ W_hy + b_y

[t=1] h_prev = h_0 (이전 hidden state가 전달됨!)
  z_h = x_1 @ W_xh + h_0 @ W_hh + b_h
  h_1 = tanh(z_h)
  y_1 = h_1 @ W_hy + b_y

[t=2] h_prev = h_1
  z_h = x_2 @ W_xh + h_1 @ W_hh + b_h
  h_2 = tanh(z_h)
  y_2 = h_2 @ W_hy + b_y

핵심: h가 시간에 따라 전달되면서 "기억"을 축적한다!
"""

print("\n" + "=" * 60)
print("Forward Pass 상세")
print("=" * 60)

h_prev = np.zeros(hidden_size)  # 초기 hidden state
hidden_states = [h_prev.copy()]
outputs = []

for t in range(len(X_seq)):
    x_t = X_seq[t]

    # RNN 핵심 수식
    z_h = x_t @ W_xh + h_prev @ W_hh + b_h
    h_t = np.tanh(z_h)

    # 출력
    y_t = h_t @ W_hy + b_y

    print(f"\n--- t={t} ---")
    print(f"  x_t     = {x_t}")
    print(f"  h_prev  = {np.round(h_prev, 4)}")
    print(f"  z_h     = x_t@W_xh + h_prev@W_hh + b_h")
    print(f"          = {np.round(x_t @ W_xh, 4)} + {np.round(h_prev @ W_hh, 4)} + {b_h}")
    print(f"          = {np.round(z_h, 4)}")
    print(f"  h_t     = tanh(z_h) = {np.round(h_t, 4)}")
    print(f"  y_t     = h_t@W_hy + b_y = {np.round(y_t, 4)}")

    h_prev = h_t
    hidden_states.append(h_t.copy())
    outputs.append(y_t.copy())

# ==============================================
# 3단계: Vanishing Gradient 문제 시각화
# ==============================================
print("\n" + "=" * 60)
print("Vanishing Gradient 문제")
print("=" * 60)

print("""
긴 시퀀스에서 gradient가 어떻게 변하는지 확인합니다.

tanh의 미분 최대값 = 1 (x=0일 때)
→ BPTT에서 gradient = Π(tanh' * W_hh) 가 계속 곱해짐
→ 시퀀스가 길면 gradient → 0 (Vanishing) 또는 → ∞ (Exploding)
""")

# 긴 시퀀스에서 gradient 크기 추적
seq_length = 50
gradients = []
h = np.zeros(hidden_size)

X_long = np.random.randn(seq_length, input_size) * 0.5
hidden_long = []

for t in range(seq_length):
    z = X_long[t] @ W_xh + h @ W_hh + b_h
    h = np.tanh(z)
    hidden_long.append(h.copy())
    # tanh의 미분 (1 - tanh^2)
    grad_tanh = 1 - h**2
    gradients.append(np.mean(np.abs(grad_tanh)))

print(f"Gradient 크기 변화:")
print(f"  t=0:  {gradients[0]:.4f}")
print(f"  t=10: {gradients[10]:.4f}")
print(f"  t=25: {gradients[25]:.4f}")
print(f"  t=49: {gradients[49]:.4f}")
print(f"\n→ Gradient가 점점 작아지면 초반 정보를 잊어버림!")
print(f"→ 해결책: LSTM (Gate로 gradient 흐름 제어)")

# ==============================================
# 시각화
# ==============================================
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Hidden state 변화
    hs = np.array(hidden_states[1:])  # (3, 3)
    for i in range(hidden_size):
        axes[0].plot(range(3), hs[:, i], 'o-', label=f'h[{i}]')
    axes[0].set_xlabel('Timestep')
    axes[0].set_ylabel('Hidden State Value')
    axes[0].set_title('Hidden State 변화 (3 timesteps)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Vanishing gradient
    axes[1].plot(gradients, 'r-')
    axes[1].set_xlabel('Timestep')
    axes[1].set_ylabel('Mean |grad(tanh)|')
    axes[1].set_title('Vanishing Gradient (50 timesteps)')
    axes[1].grid(True, alpha=0.3)

    # Hidden state norm over long sequence
    h_norms = [np.linalg.norm(h) for h in hidden_long]
    axes[2].plot(h_norms, 'b-')
    axes[2].set_xlabel('Timestep')
    axes[2].set_ylabel('||h||')
    axes[2].set_title('Hidden State Norm 변화')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('C:/Users/Ryan/AI5/curriculum/03_RNN/rnn_analysis.png', dpi=100)
    plt.show()
except ImportError:
    pass

# ==============================================
# ★ 과제 ★
# ==============================================
"""
[손계산 과제]
1. W_xh=[[0.1, 0.2], [0.3, 0.4]], W_hh=[[0.5, 0.6], [0.7, 0.8]], b_h=[0,0]
   h_0=[0,0], 입력 시퀀스 x=[[1, 0], [0, 1], [1, 1]]
   → 3 timestep의 h_t를 종이에 계산하세요.

2. 위 계산에서 tanh 미분값을 각 timestep마다 구하고,
   t=2의 gradient가 t=0까지 전파될 때 얼마나 줄어드는지 계산하세요.

[코딩 과제]
3. 간단한 시계열 (sin 함수)를 RNN으로 예측하는 코드를 작성하세요.
4. 시퀀스 길이를 10, 50, 100으로 바꿔가며 학습 성능을 비교하세요.
"""
