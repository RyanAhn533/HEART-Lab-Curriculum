"""
========================================
07-01. Transformer - Self Attention numpy 손코딩
========================================
"Attention is All You Need" (2017)

RNN/LSTM 없이 Attention만으로 모든 것을 처리!

이 파일에서 배우는 것:
1. Self-Attention: 같은 시퀀스 내에서 서로 참조
2. Multi-Head Attention: 여러 관점에서 동시에 보기
3. Positional Encoding: 순서 정보 추가
4. Transformer Block 전체 구조

★ Q, K, V 행렬곱을 3 token에 대해 종이에 전개 ★
"""

import numpy as np

np.random.seed(42)

# ==============================================
# 1단계: Self-Attention
# ==============================================
print("=" * 60)
print("Self-Attention 손계산")
print("=" * 60)
print("""
문장: "나는 학생이다"
Self-Attention은 각 단어가 다른 모든 단어와의 관계를 계산합니다.

"학생" → "나는"과 관련? "이다"와 관련?
→ 모든 단어 쌍의 관련도를 동시에 계산!

RNN과의 차이:
  RNN:  순차적으로 처리 (t=0 → t=1 → t=2)
  Self-Attention: 모든 위치를 동시에 처리 (병렬화!)
""")

# 입력: 3개 토큰, 각 4차원 임베딩
# "나는"=token0, "학생"=token1, "이다"=token2
X = np.array([
    [1.0, 0.5, 0.3, 0.2],   # "나는"
    [0.2, 0.8, 0.7, 0.1],   # "학생"
    [0.5, 0.3, 0.1, 0.9],   # "이다"
])

d_model = 4   # 임베딩 차원
d_k = 3       # Q, K 차원 (보통 d_model // num_heads)
d_v = 3       # V 차원

# Q, K, V를 만드는 가중치 행렬 (학습되는 파라미터)
W_Q = np.random.randn(d_model, d_k) * 0.5
W_K = np.random.randn(d_model, d_k) * 0.5
W_V = np.random.randn(d_model, d_v) * 0.5

print(f"입력 X ({X.shape}):")
for i, word in enumerate(["나는", "학생", "이다"]):
    print(f"  {word}: {X[i]}")

# ==============================================
# 2단계: Q, K, V 계산
# ==============================================
"""
★★★ Self-Attention 수식 ★★★

1. Q = X @ W_Q   (각 토큰이 "무엇을 찾고 싶은지")
2. K = X @ W_K   (각 토큰이 "무엇을 갖고 있는지")
3. V = X @ W_V   (각 토큰의 "실제 정보")

4. Score = Q @ K^T / √d_k
5. Attention Weight = softmax(Score)
6. Output = Weight @ V
"""

Q = X @ W_Q  # (3, 3)
K = X @ W_K  # (3, 3)
V = X @ W_V  # (3, 3)

print(f"\nQ (Query) = X @ W_Q:")
for i, word in enumerate(["나는", "학생", "이다"]):
    print(f"  {word}: {np.round(Q[i], 4)}")

print(f"\nK (Key) = X @ W_K:")
for i, word in enumerate(["나는", "학생", "이다"]):
    print(f"  {word}: {np.round(K[i], 4)}")

print(f"\nV (Value) = X @ W_V:")
for i, word in enumerate(["나는", "학생", "이다"]):
    print(f"  {word}: {np.round(V[i], 4)}")

# ==============================================
# 3단계: Attention Score & Weight
# ==============================================
def softmax(z, axis=-1):
    e = np.exp(z - np.max(z, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

# Score = Q @ K^T / √d_k
scores = Q @ K.T / np.sqrt(d_k)  # (3, 3)
weights = softmax(scores)         # (3, 3)

words = ["나는", "학생", "이다"]
print(f"\nAttention Scores (Q @ K^T / √{d_k}):")
print(f"{'':>8}", end="")
for w in words:
    print(f"{w:>10}", end="")
print()
for i, w in enumerate(words):
    print(f"{w:>8}", end="")
    for j in range(3):
        print(f"{scores[i,j]:10.4f}", end="")
    print()

print(f"\nAttention Weights (softmax):")
print(f"{'':>8}", end="")
for w in words:
    print(f"{w:>10}", end="")
print()
for i, w in enumerate(words):
    print(f"{w:>8}", end="")
    for j in range(3):
        print(f"{weights[i,j]:10.4f}", end="")
    bar = " | " + " ".join(["█" * int(weights[i,j] * 15) for j in range(3)])
    print(bar)

# ==============================================
# 4단계: Output
# ==============================================
output = weights @ V  # (3, 3)

print(f"\nSelf-Attention Output = Weights @ V:")
for i, w in enumerate(words):
    print(f"  {w}: {np.round(output[i], 4)}")

print(f"\n→ 각 토큰의 출력은 다른 모든 토큰의 Value의 가중 평균!")
print(f"→ 관련 높은 토큰의 정보를 더 많이 반영")

# ==============================================
# 5단계: Multi-Head Attention
# ==============================================
print("\n" + "=" * 60)
print("Multi-Head Attention")
print("=" * 60)
print("""
Single Head: 하나의 관점으로만 관계를 봄
Multi-Head:  여러 관점(head)에서 동시에 관계를 봄

예시 (2 heads):
  Head 1: 문법적 관계에 집중 ("나는" → "이다" 주어-서술어)
  Head 2: 의미적 관계에 집중 ("나는" → "학생" 주어-보어)

구현:
  d_model = 8, num_heads = 2 → d_k = d_model // num_heads = 4
  각 head가 독립적으로 Attention 계산 → 결과를 concat → Linear
""")

num_heads = 2
d_model_mh = 4
d_head = d_model_mh // num_heads  # 각 head의 차원 = 2

head_outputs = []
for head in range(num_heads):
    # 각 head마다 독립적인 W_Q, W_K, W_V
    W_Q_h = np.random.randn(d_model_mh, d_head) * 0.5
    W_K_h = np.random.randn(d_model_mh, d_head) * 0.5
    W_V_h = np.random.randn(d_model_mh, d_head) * 0.5

    Q_h = X @ W_Q_h
    K_h = X @ W_K_h
    V_h = X @ W_V_h

    scores_h = Q_h @ K_h.T / np.sqrt(d_head)
    weights_h = softmax(scores_h)
    output_h = weights_h @ V_h

    head_outputs.append(output_h)
    print(f"  Head {head+1} weights:")
    for i, w in enumerate(words):
        print(f"    {w}: {np.round(weights_h[i], 3)}")

# Concat
multi_head_concat = np.concatenate(head_outputs, axis=-1)  # (3, 4)
print(f"\nMulti-Head concat shape: {multi_head_concat.shape}")
print(f"→ {num_heads}개 head의 출력을 이어붙임 → 다양한 관점의 정보 결합")

# ==============================================
# 6단계: Positional Encoding
# ==============================================
print("\n" + "=" * 60)
print("Positional Encoding")
print("=" * 60)
print("""
Self-Attention의 문제: 순서 정보가 없다!
  "나는 학생이다"와 "학생이다 나는" 을 구분 못함

해결: 각 위치에 고유한 벡터를 더해줌

PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
""")

def positional_encoding(max_len, d_model):
    PE = np.zeros((max_len, d_model))
    for pos in range(max_len):
        for i in range(0, d_model, 2):
            PE[pos, i] = np.sin(pos / 10000 ** (i / d_model))
            if i + 1 < d_model:
                PE[pos, i+1] = np.cos(pos / 10000 ** (i / d_model))
    return PE

PE = positional_encoding(3, d_model)
print(f"Positional Encoding (3 positions, {d_model}-dim):")
for pos in range(3):
    print(f"  pos={pos}: {np.round(PE[pos], 4)}")

X_with_pos = X + PE
print(f"\n입력 + Positional Encoding:")
for i, w in enumerate(words):
    print(f"  {w}: {np.round(X[i], 4)} + {np.round(PE[i], 4)} = {np.round(X_with_pos[i], 4)}")

# ==============================================
# 7단계: Transformer Block 전체
# ==============================================
print("\n" + "=" * 60)
print("Transformer Block 전체 구조")
print("=" * 60)
print("""
┌─────────────────────────────┐
│  Input Embedding            │
│  + Positional Encoding      │
├─────────────────────────────┤
│  Multi-Head Self-Attention  │
│  + Add & LayerNorm          │  ← Residual Connection
├─────────────────────────────┤
│  Feed-Forward Network       │
│  (Linear → ReLU → Linear)  │
│  + Add & LayerNorm          │  ← Residual Connection
├─────────────────────────────┤
│  Output                     │
└─────────────────────────────┘

이것이 N번 반복됩니다 (BERT: N=12, GPT-3: N=96)
""")

# 간단한 Transformer Block 구현
def layer_norm(x, eps=1e-6):
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + eps)

def feed_forward(x, W1, b1, W2, b2):
    return np.maximum(0, x @ W1 + b1) @ W2 + b2  # ReLU activation

# FFN 가중치
d_ff = 8  # Feed-forward 내부 차원 (보통 d_model * 4)
W_ff1 = np.random.randn(d_v, d_ff) * 0.1
b_ff1 = np.zeros(d_ff)
W_ff2 = np.random.randn(d_ff, d_v) * 0.1
b_ff2 = np.zeros(d_v)

# Transformer Block
attn_output = output  # Self-Attention 출력 (이미 계산됨, shape: 3x3)

# 1) Add & Norm (Residual)
# 참고: X와 attn_output 차원이 다를 수 있으므로 V를 사용
residual1 = V + attn_output  # Residual connection
norm1 = layer_norm(residual1)

# 2) Feed-Forward
ff_output = feed_forward(norm1, W_ff1, b_ff1, W_ff2, b_ff2)

# 3) Add & Norm
residual2 = norm1 + ff_output
norm2 = layer_norm(residual2)

print(f"Transformer Block 출력:")
for i, w in enumerate(words):
    print(f"  {w}: {np.round(norm2[i], 4)}")

# ==============================================
# 시각화
# ==============================================
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Attention heatmap
    im = axes[0].imshow(weights, cmap='Blues', vmin=0, vmax=1)
    axes[0].set_xticks(range(3))
    axes[0].set_xticklabels(words, fontsize=12)
    axes[0].set_yticks(range(3))
    axes[0].set_yticklabels(words, fontsize=12)
    axes[0].set_title('Self-Attention Weights', fontsize=13)
    for i in range(3):
        for j in range(3):
            axes[0].text(j, i, f'{weights[i,j]:.2f}', ha='center', va='center', fontsize=12)
    plt.colorbar(im, ax=axes[0])

    # Positional Encoding
    PE_vis = positional_encoding(50, 32)
    axes[1].imshow(PE_vis, cmap='RdBu', aspect='auto')
    axes[1].set_xlabel('Dimension')
    axes[1].set_ylabel('Position')
    axes[1].set_title('Positional Encoding (50 pos, 32-dim)', fontsize=13)

    # Multi-head comparison
    for h_idx, h_out in enumerate(head_outputs):
        axes[2].bar(np.arange(3) + h_idx * 0.35, np.linalg.norm(h_out, axis=1),
                    width=0.35, label=f'Head {h_idx+1}')
    axes[2].set_xticks(range(3))
    axes[2].set_xticklabels(words, fontsize=12)
    axes[2].set_ylabel('Output Norm')
    axes[2].set_title('Multi-Head Output 비교', fontsize=13)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('C:/Users/Ryan/AI5/curriculum/07_Transformer/transformer_analysis.png', dpi=100)
    plt.show()
except ImportError:
    pass

# ==============================================
# ★ 과제 ★
# ==============================================
"""
[손계산 과제 - 필수!!!]
1. X = [[1, 0], [0, 1], [1, 1]], W_Q = W_K = W_V = I (단위행렬)
   → Q, K, V, Scores, Weights, Output 전체를 종이에 계산하세요.

2. Scaling (√d_k)을 하는 이유: d_k=64일 때 내적값의 평균/분산을 계산하고
   softmax에 미치는 영향을 설명하세요.

3. Positional Encoding에서 sin/cos을 쓰는 이유:
   PE(pos+k)가 PE(pos)의 선형변환으로 표현됨을 보이세요.

[코딩 과제]
4. Transformer Encoder Block을 PyTorch nn.Module로 구현하세요.
5. nn.TransformerEncoder와 결과를 비교하세요.
"""
