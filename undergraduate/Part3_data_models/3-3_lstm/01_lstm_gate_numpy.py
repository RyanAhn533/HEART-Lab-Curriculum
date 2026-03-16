"""
========================================
04-01. LSTM Gate 동작 원리 - numpy 손코딩
========================================
RNN의 Vanishing Gradient 문제를 해결하기 위해 등장

LSTM의 핵심: 3개의 Gate + Cell State
  - Forget Gate:  이전 기억 중 얼마나 버릴지 (0~1)
  - Input Gate:   새 정보를 얼마나 받아들일지 (0~1)
  - Output Gate:  cell state에서 얼마나 내보낼지 (0~1)
  - Cell State:   장기 기억 (고속도로처럼 gradient가 잘 흐름)

★ 각 gate의 출력을 3 timestep 동안 종이에 계산하세요 ★
"""

import numpy as np

np.random.seed(42)

# ==============================================
# 1단계: LSTM 셀 하나 구현
# ==============================================
print("=" * 60)
print("LSTM Cell 동작 원리")
print("=" * 60)

input_size = 2
hidden_size = 3

# 가중치 (실제로는 4배 크기로 한번에 계산하지만, 교육용으로 분리)
# Forget Gate
W_f = np.random.randn(input_size + hidden_size, hidden_size) * 0.1
b_f = np.zeros(hidden_size)

# Input Gate
W_i = np.random.randn(input_size + hidden_size, hidden_size) * 0.1
b_i = np.zeros(hidden_size)

# Cell candidate
W_c = np.random.randn(input_size + hidden_size, hidden_size) * 0.1
b_c = np.zeros(hidden_size)

# Output Gate
W_o = np.random.randn(input_size + hidden_size, hidden_size) * 0.1
b_o = np.zeros(hidden_size)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 입력 시퀀스
X_seq = np.array([
    [1.0, 0.5],   # t=0
    [0.3, 0.8],   # t=1
    [0.7, 0.2],   # t=2
])

# ==============================================
# 2단계: Forward Pass 상세
# ==============================================
"""
★★★ LSTM 수식 (종이에 따라 써보세요) ★★★

입력: x_t (현재 입력), h_(t-1) (이전 hidden), c_(t-1) (이전 cell)
concat = [x_t, h_(t-1)]  ← 입력과 이전 hidden을 합침

1) Forget Gate:  f_t = sigmoid(concat @ W_f + b_f)
   → f_t가 0이면 이전 기억 완전 삭제, 1이면 완전 보존

2) Input Gate:   i_t = sigmoid(concat @ W_i + b_i)
   Cell Candidate: c_tilde = tanh(concat @ W_c + b_c)
   → 새로 기억할 후보 정보

3) Cell Update:  c_t = f_t * c_(t-1) + i_t * c_tilde
   → 이전 기억 일부 삭제 + 새 정보 일부 추가

4) Output Gate:  o_t = sigmoid(concat @ W_o + b_o)
   Hidden State:  h_t = o_t * tanh(c_t)
   → cell state에서 필요한 부분만 출력
"""

h_prev = np.zeros(hidden_size)
c_prev = np.zeros(hidden_size)

print("\n각 Gate의 동작을 timestep별로 확인:")

all_gates = {'forget': [], 'input': [], 'output': [], 'cell': [], 'hidden': []}

for t in range(len(X_seq)):
    x_t = X_seq[t]
    concat = np.concatenate([x_t, h_prev])  # [x_t, h_(t-1)]

    # 1) Forget Gate
    f_t = sigmoid(concat @ W_f + b_f)

    # 2) Input Gate + Cell Candidate
    i_t = sigmoid(concat @ W_i + b_i)
    c_tilde = np.tanh(concat @ W_c + b_c)

    # 3) Cell State Update
    c_t = f_t * c_prev + i_t * c_tilde

    # 4) Output Gate + Hidden State
    o_t = sigmoid(concat @ W_o + b_o)
    h_t = o_t * np.tanh(c_t)

    # 기록
    all_gates['forget'].append(f_t.copy())
    all_gates['input'].append(i_t.copy())
    all_gates['output'].append(o_t.copy())
    all_gates['cell'].append(c_t.copy())
    all_gates['hidden'].append(h_t.copy())

    print(f"\n{'='*40} t={t} {'='*40}")
    print(f"  x_t      = {x_t}")
    print(f"  h_prev   = {np.round(h_prev, 4)}")
    print(f"  c_prev   = {np.round(c_prev, 4)}")
    print(f"  concat   = {np.round(concat, 4)}")
    print(f"  ----- Gates -----")
    print(f"  f_t (forget) = {np.round(f_t, 4)}  ← 이전 기억 보존 비율")
    print(f"  i_t (input)  = {np.round(i_t, 4)}  ← 새 정보 수용 비율")
    print(f"  c_tilde      = {np.round(c_tilde, 4)}  ← 새 정보 후보")
    print(f"  o_t (output) = {np.round(o_t, 4)}  ← 출력 비율")
    print(f"  ----- State Update -----")
    print(f"  c_t = f*c_prev + i*c_tilde")
    print(f"      = {np.round(f_t*c_prev, 4)} + {np.round(i_t*c_tilde, 4)}")
    print(f"      = {np.round(c_t, 4)}  ← 새 cell state (장기 기억)")
    print(f"  h_t = o * tanh(c_t)")
    print(f"      = {np.round(o_t, 4)} * {np.round(np.tanh(c_t), 4)}")
    print(f"      = {np.round(h_t, 4)}  ← 새 hidden state (출력)")

    h_prev = h_t
    c_prev = c_t

# ==============================================
# 3단계: RNN vs LSTM Gradient 비교
# ==============================================
print("\n" + "=" * 60)
print("왜 LSTM이 Vanishing Gradient를 해결하는가?")
print("=" * 60)
print("""
RNN:  gradient = Π(tanh' * W_hh)  → 계속 곱해져서 0으로 소멸

LSTM: cell state gradient = Π(f_t)
      → f_t는 sigmoid 출력 (0~1)
      → 학습을 통해 f_t ≈ 1 로 만들 수 있음
      → gradient가 거의 그대로 전달됨 (고속도로!)

비유:
  RNN = 구불구불한 산길 (gradient가 마찰로 소멸)
  LSTM = 고속도로 + IC (cell state는 쭉 흐르고, gate가 진입/이탈 제어)
""")

# 수치 비교
print("50 timestep 후 gradient 크기 비교:")
seq_len = 50

# RNN gradient decay
rnn_grad = 1.0
for t in range(seq_len):
    rnn_grad *= 0.7  # tanh' 평균 ~0.7

# LSTM gradient decay (forget gate ~0.95로 가정)
lstm_grad = 1.0
for t in range(seq_len):
    lstm_grad *= 0.95  # forget gate가 학습됨

print(f"  RNN  gradient: {rnn_grad:.10f} (사실상 0)")
print(f"  LSTM gradient: {lstm_grad:.4f}       (정보 유지!)")

# ==============================================
# 시각화
# ==============================================
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    gate_names = ['forget', 'input', 'output', 'cell']
    titles = ['Forget Gate (기억 보존)', 'Input Gate (새 정보 수용)',
              'Output Gate (출력 비율)', 'Cell State (장기 기억)']

    for idx, (gate, title) in enumerate(zip(gate_names, titles)):
        ax = axes[idx // 2][idx % 2]
        data = np.array(all_gates[gate])
        for i in range(hidden_size):
            ax.plot(range(3), data[:, i], 'o-', label=f'dim[{i}]', markersize=8)
        ax.set_xlabel('Timestep')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks([0, 1, 2])

    plt.suptitle('LSTM Gate 동작 시각화', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('C:/Users/Ryan/AI5/curriculum/04_LSTM/lstm_gates.png', dpi=100)
    plt.show()
except ImportError:
    pass

# ==============================================
# ★ 과제 ★
# ==============================================
"""
[손계산 과제 - 반드시 해야 함]
1. hidden_size=2로 줄이고, 입력 [1, 0]에 대해:
   concat, f_t, i_t, c_tilde, c_t, o_t, h_t 를 모두 계산하세요.
   (W는 코드에서 출력한 값 사용)

2. f_t=0이면 어떻게 되는가? f_t=1이면? 직관적으로 설명하세요.

3. RNN에서 50 timestep 후 gradient 크기를 계산하세요.
   (tanh 미분 최대 = 1, 평균 ≈ 0.65 가정)

[코딩 과제]
4. Keras의 LSTM layer와 위 코드의 결과를 비교하세요.
5. 긴 시퀀스 (100 timestep) sin 함수 예측을 RNN vs LSTM으로 비교하세요.
"""
