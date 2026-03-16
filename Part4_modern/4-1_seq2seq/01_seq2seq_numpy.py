"""
========================================
05-01. Seq2Seq (Encoder-Decoder) - 개념과 구현
========================================
LSTM/RNN의 한계: 입력과 출력 길이가 같아야 함
→ 번역: "나는 학생이다" (3단어) → "I am a student" (4단어) → 길이 다름!

해결: Encoder-Decoder 구조
  Encoder: 입력 시퀀스 → 하나의 벡터(Context Vector)로 압축
  Decoder: Context Vector → 출력 시퀀스 생성

이 파일에서 배우는 것:
1. Encoder가 시퀀스를 어떻게 압축하는지
2. Context Vector의 역할
3. Decoder가 어떻게 한 단어씩 생성하는지
4. Teacher Forcing

★ Encoder 3 timestep → Context → Decoder 4 timestep 을 종이에 그려보세요 ★
"""

import numpy as np

np.random.seed(42)

# ==============================================
# 1단계: 간단한 숫자 뒤집기 태스크
# ==============================================
print("=" * 60)
print("Seq2Seq 개념: 숫자 시퀀스 뒤집기")
print("=" * 60)
print("""
태스크: [1, 2, 3] → [3, 2, 1]
       입력 길이와 출력 길이가 같지만,
       Seq2Seq 구조를 이해하기 위한 간단한 예시입니다.

실제 응용:
  - 번역: "나는 학생이다" → "I am a student"
  - 요약: 긴 문장 → 짧은 문장
  - 챗봇: 질문 → 답변
""")

# ==============================================
# 2단계: Encoder
# ==============================================
print("\n" + "=" * 60)
print("Encoder: 입력 시퀀스 → Context Vector")
print("=" * 60)

input_size = 4   # 0, 1, 2, 3 → one-hot encoding
hidden_size = 5

# One-hot encoding
def one_hot(idx, size):
    vec = np.zeros(size)
    vec[idx] = 1.0
    return vec

# Encoder 가중치
W_enc_xh = np.random.randn(input_size, hidden_size) * 0.3
W_enc_hh = np.random.randn(hidden_size, hidden_size) * 0.3
b_enc = np.zeros(hidden_size)

# 입력: [1, 2, 3]
input_seq = [1, 2, 3]
print(f"입력 시퀀스: {input_seq}")

h_enc = np.zeros(hidden_size)
encoder_states = []

for t, token in enumerate(input_seq):
    x_t = one_hot(token, input_size)
    z = x_t @ W_enc_xh + h_enc @ W_enc_hh + b_enc
    h_enc = np.tanh(z)
    encoder_states.append(h_enc.copy())

    print(f"\n  Encoder t={t}: input={token}")
    print(f"    x_t (one-hot) = {x_t}")
    print(f"    h_enc = {np.round(h_enc, 4)}")

# Context Vector = Encoder의 마지막 hidden state
context_vector = h_enc
print(f"\n★ Context Vector = {np.round(context_vector, 4)}")
print(f"  → 입력 시퀀스 [1,2,3]의 정보가 이 벡터 하나에 압축됨!")

# ==============================================
# 3단계: Decoder
# ==============================================
print("\n" + "=" * 60)
print("Decoder: Context Vector → 출력 시퀀스")
print("=" * 60)

# Decoder 가중치
W_dec_xh = np.random.randn(input_size, hidden_size) * 0.3
W_dec_hh = np.random.randn(hidden_size, hidden_size) * 0.3
b_dec = np.zeros(hidden_size)

W_out = np.random.randn(hidden_size, input_size) * 0.3
b_out = np.zeros(input_size)

def softmax(z):
    e = np.exp(z - np.max(z))
    return e / e.sum()

# Decoder 시작: h_dec = context_vector
h_dec = context_vector.copy()
target_seq = [3, 2, 1]  # 뒤집은 정답
SOS = 0  # Start of Sequence token

print(f"목표 출력: {target_seq}")
print(f"Decoder 초기 hidden = Context Vector")

# Teacher Forcing으로 디코딩
prev_token = SOS
for t in range(len(target_seq)):
    x_t = one_hot(prev_token, input_size)
    z = x_t @ W_dec_xh + h_dec @ W_dec_hh + b_dec
    h_dec = np.tanh(z)

    # 출력 계산
    logits = h_dec @ W_out + b_out
    probs = softmax(logits)
    predicted = np.argmax(probs)

    print(f"\n  Decoder t={t}:")
    print(f"    입력 token = {prev_token} (Teacher Forcing: 이전 정답 사용)")
    print(f"    h_dec = {np.round(h_dec, 4)}")
    print(f"    출력 확률 = {np.round(probs, 3)} → 예측: {predicted} (정답: {target_seq[t]})")

    # Teacher Forcing: 학습 시에는 정답을 다음 입력으로 사용
    prev_token = target_seq[t]

# ==============================================
# 4단계: Seq2Seq의 한계
# ==============================================
print("\n" + "=" * 60)
print("Seq2Seq의 한계 → Attention의 필요성")
print("=" * 60)
print("""
문제: Context Vector 병목 (Information Bottleneck)

"나는 어제 학교에서 친구와 점심을 먹고 도서관에서 공부를 했다"
  → 이 긴 문장의 모든 정보를 벡터 하나에 우겨넣어야 함
  → 문장이 길수록 앞부분 정보 손실

해결: Attention Mechanism
  → Decoder가 매 timestep마다 Encoder의 모든 hidden state를 참조
  → "지금 어디를 봐야 하는지" 가중치를 동적으로 계산

┌─────────┐                     ┌─────────┐
│ Encoder │─── Context만 ──────→│ Decoder │  ← Seq2Seq (병목!)
└─────────┘                     └─────────┘

┌─────────┐── h1 ──┐
│ Encoder │── h2 ──┼── Attention ──→│ Decoder │  ← + Attention
└─────────┘── h3 ──┘                └─────────┘
               ↑ 모든 hidden state 참조
""")

# ==============================================
# ★ 과제 ★
# ==============================================
"""
[손계산 과제]
1. 입력 [2, 1]에 대해 Encoder의 각 timestep hidden state를 계산하세요.
2. Context Vector에서 시작하여 Decoder 2 timestep의 출력을 계산하세요.
3. 입력이 [1,2,3,4,5,6,7,8,9,10]처럼 길어지면 Context Vector 하나로
   충분한지 논의하세요.

[코딩 과제]
4. Keras로 Seq2Seq 번역 모델을 구현하세요. (영어 단어 → 한국어 단어)
5. Teacher Forcing 없이 (자기 예측을 다음 입력으로) 학습하면 어떻게 되나요?
"""
