"""
========================================
06-01. Attention Mechanism - numpy 손코딩
========================================
Seq2Seq의 병목 해결: Decoder가 Encoder의 모든 timestep을 참조

핵심 아이디어:
  "번역할 때 매 단어마다 원문의 어디를 봐야 하는지 가중치를 동적으로 계산"

이 파일에서 배우는 것:
1. Attention Score 계산 (Dot-product)
2. Attention Weight (Softmax)
3. Context Vector 계산
4. Bahdanau (Additive) vs Luong (Dot-product) Attention

★ 3개의 Encoder hidden state에 대한 Attention을 종이에 계산 ★
"""

import numpy as np

np.random.seed(42)

# ==============================================
# 1단계: 가장 직관적인 Attention
# ==============================================
print("=" * 60)
print("Attention이란?")
print("=" * 60)
print("""
번역 예시: "나는 학생이다" → "I am a student"

Seq2Seq (Attention 없음):
  Encoder: "나는" → h1, "학생" → h2, "이다" → h3
  Context = h3 만 사용 (병목!)

Seq2Seq + Attention:
  "I" 생성할 때:  "나는"(h1)에 집중 → attention = [0.8, 0.1, 0.1]
  "am" 생성할 때:  "이다"(h3)에 집중 → attention = [0.1, 0.1, 0.8]
  "student" 생성할 때: "학생"(h2)에 집중 → attention = [0.1, 0.8, 0.1]
""")

# ==============================================
# 2단계: Dot-Product Attention 손계산
# ==============================================
print("=" * 60)
print("Dot-Product Attention 손계산")
print("=" * 60)

# Encoder hidden states (3 timesteps, 각 4차원)
encoder_outputs = np.array([
    [0.5, 0.1, -0.2, 0.8],   # h1: "나는"
    [-0.1, 0.7, 0.3, 0.2],   # h2: "학생"
    [0.3, 0.2, 0.6, -0.1],   # h3: "이다"
])

# Decoder hidden state (현재 timestep)
decoder_hidden = np.array([0.4, 0.3, -0.1, 0.6])  # "I"를 생성하려는 중

def softmax(z):
    e = np.exp(z - np.max(z))
    return e / e.sum()

print(f"Encoder outputs (3 timesteps, 4-dim):")
for i, h in enumerate(encoder_outputs):
    print(f"  h{i+1} = {h}")
print(f"\nDecoder hidden (현재): {decoder_hidden}")

"""
★★★ Attention 손계산 가이드 ★★★

Step 1: Attention Score (유사도 계산)
  score_i = decoder_hidden · encoder_output_i  (내적)

Step 2: Attention Weight (정규화)
  weight_i = softmax(scores)  → 합이 1이 됨

Step 3: Context Vector (가중 평균)
  context = Σ(weight_i * encoder_output_i)
"""

# Step 1: Score 계산 (내적)
scores = np.array([np.dot(decoder_hidden, h) for h in encoder_outputs])
print(f"\nStep 1. Attention Scores (내적):")
for i in range(3):
    dot_detail = " + ".join([f"{decoder_hidden[j]:.1f}*{encoder_outputs[i,j]:.1f}" for j in range(4)])
    print(f"  score({i+1}) = decoder · h{i+1} = {dot_detail} = {scores[i]:.4f}")

# Step 2: Softmax → Weight
weights = softmax(scores)
print(f"\nStep 2. Attention Weights (softmax):")
for i in range(3):
    print(f"  weight({i+1}) = {weights[i]:.4f}  {'← 가장 집중!' if weights[i] == max(weights) else ''}")
print(f"  합계 = {sum(weights):.4f} (반드시 1)")

# Step 3: Context Vector
context = np.sum(weights.reshape(-1, 1) * encoder_outputs, axis=0)
print(f"\nStep 3. Context Vector (가중 평균):")
print(f"  context = {weights[0]:.4f}*h1 + {weights[1]:.4f}*h2 + {weights[2]:.4f}*h3")
print(f"  context = {np.round(context, 4)}")

# ==============================================
# 3단계: Scaled Dot-Product Attention (Transformer에서 사용)
# ==============================================
print("\n" + "=" * 60)
print("Scaled Dot-Product Attention (Q, K, V)")
print("=" * 60)
print("""
Transformer의 Attention:
  Q (Query):  "내가 찾고 싶은 것" (Decoder hidden)
  K (Key):    "검색 키" (Encoder hidden)
  V (Value):  "실제 값" (Encoder hidden)

  Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
                              ↑ scaling (값이 너무 커지는 것 방지)
""")

d_k = 4  # key 차원

# Q, K, V 설정
Q = decoder_hidden.reshape(1, -1)        # (1, 4)
K = encoder_outputs                       # (3, 4)
V = encoder_outputs                       # (3, 4)

# Scaled Dot-Product
scores_scaled = (Q @ K.T) / np.sqrt(d_k)  # (1, 3)
weights_scaled = softmax(scores_scaled[0])
context_scaled = weights_scaled @ V        # (1, 4)

print(f"Q (Query) = {Q[0]}")
print(f"K (Keys)  = encoder_outputs")
print(f"V (Values) = encoder_outputs")
print(f"\nscores = Q·K^T / √{d_k} = {np.round(scores_scaled[0], 4)}")
print(f"weights = softmax(scores) = {np.round(weights_scaled, 4)}")
print(f"context = weights · V     = {np.round(context_scaled[0], 4)}")

# ==============================================
# 4단계: Attention 시각화
# ==============================================
print("\n" + "=" * 60)
print("Attention Weight 해석")
print("=" * 60)

# 여러 decoder timestep에 대한 attention
decoder_states = np.array([
    [0.4, 0.3, -0.1, 0.6],   # "I" 생성 시
    [-0.2, 0.6, 0.4, 0.1],   # "am" 생성 시
    [0.1, 0.8, 0.2, 0.3],    # "a" 생성 시
    [-0.1, 0.5, 0.3, 0.4],   # "student" 생성 시
])

source_words = ["나는", "학생", "이다"]
target_words = ["I", "am", "a", "student"]

print(f"\n번역 Attention Matrix:")
print(f"{'':>10}", end="")
for w in source_words:
    print(f"{w:>10}", end="")
print()

attention_matrix = []
for t, (dec_h, target_w) in enumerate(zip(decoder_states, target_words)):
    scores = np.array([np.dot(dec_h, h) for h in encoder_outputs])
    weights = softmax(scores)
    attention_matrix.append(weights)

    print(f"{target_w:>10}", end="")
    for w in weights:
        bar = "█" * int(w * 20)
        print(f"  {w:.3f}{bar}", end="")
    print()

try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    am = np.array(attention_matrix)
    im = ax.imshow(am, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(source_words)))
    ax.set_xticklabels(source_words, fontsize=14)
    ax.set_yticks(range(len(target_words)))
    ax.set_yticklabels(target_words, fontsize=14)
    ax.set_xlabel('Source (Encoder)', fontsize=12)
    ax.set_ylabel('Target (Decoder)', fontsize=12)
    ax.set_title('Attention Weights Heatmap', fontsize=14)

    for i in range(len(target_words)):
        for j in range(len(source_words)):
            ax.text(j, i, f'{am[i,j]:.2f}', ha='center', va='center', fontsize=12)

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig('C:/Users/Ryan/AI5/curriculum/06_Attention/attention_heatmap.png', dpi=100)
    plt.show()
except ImportError:
    pass

# ==============================================
# ★ 과제 ★
# ==============================================
"""
[손계산 과제]
1. Encoder outputs = [[1, 0], [0, 1], [1, 1]], Decoder hidden = [0.5, 0.8]
   → Dot-product attention의 scores, weights, context를 계산하세요.

2. √d_k로 나누는 Scaling을 하면/안하면 softmax 결과가 어떻게 달라지나요?
   d_k=64일 때 score=[50, 10, 5]에 대해 직접 계산해보세요.

3. Bahdanau Attention: score = V^T * tanh(W1*h_enc + W2*h_dec)
   위와 같은 데이터로 계산하세요. (W1, W2, V는 임의의 작은 값)

[코딩 과제]
4. 위 번역 예시를 Keras로 Seq2Seq + Attention 모델로 구현하세요.
5. Attention weight를 시각화하여 모델이 어디를 보는지 분석하세요.
"""
