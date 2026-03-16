# 손코딩 연습 문제

종이와 펜만 가지고 푸세요. 계산기/코드 금지.
풀고 나서 numpy 코드로 검증하세요.

---

## Level 1: 선형회귀

### 문제 1-1: 기본 Forward + Backward
데이터: x=[1, 2, 3], y=[2, 4, 6] (정답: y=2x)
초기값: w=0, b=0, lr=0.01

**3 epoch를 계산하세요:**
| Epoch | pred (3개) | MSE Loss | dw | db | 새 w | 새 b |
|-------|-----------|----------|----|----|------|------|
| 1 | | | | | | |
| 2 | | | | | | |
| 3 | | | | | | |

### 문제 1-2: Learning Rate 실험
같은 데이터, w=0, b=0에서 lr=0.1로 2 epoch 계산.
lr=0.01과 비교하여 Loss 감소 속도가 어떻게 다른가?

### 문제 1-3: 발산
lr=1.0으로 하면 어떻게 되는가? 2 epoch만 계산해보세요.

---

## Level 2: MLP 역전파

### 문제 2-1: Forward Pass
네트워크: Input(2) → Hidden(2, sigmoid) → Output(1, sigmoid)
W1=[[0.5, -0.3], [0.2, 0.4]], b1=[0, 0]
W2=[[0.6], [-0.5]], b2=[0]
입력: x=[1, 0], 정답: y=1

계산하세요:
- z1 = x @ W1 + b1 = ?
- a1 = sigmoid(z1) = ?
- z2 = a1 @ W2 + b2 = ?
- a2 = sigmoid(z2) = ? (이게 예측값)
- Loss = (a2 - y)² = ?

### 문제 2-2: Backward Pass (Chain Rule)
위 결과에서 역전파를 계산하세요:

Output layer:
- delta2 = (a2 - y) × sigmoid'(a2) = ?
- dW2 = a1^T × delta2 = ?
- db2 = delta2 = ?

Hidden layer:
- delta1 = (delta2 × W2^T) × sigmoid'(a1) = ?
- dW1 = x^T × delta1 = ?
- db1 = delta1 = ?

### 문제 2-3: Weight Update
lr=0.5로 모든 가중치를 업데이트하세요:
- W1_new = W1 - lr × dW1 = ?
- W2_new = W2 - lr × dW2 = ?

이걸 3 epoch 반복하세요.

---

## Level 3: CNN

### 문제 3-1: Convolution
입력 (4×4):
```
1 2 0 1
0 1 2 3
1 0 1 0
2 1 0 1
```
필터 (3×3):
```
1  0 -1
1  0 -1
1  0 -1
```
stride=1, padding=0일 때 출력을 계산하세요. (출력 크기: ?×?)

### 문제 3-2: 출력 크기 공식
공식: (Input - Kernel + 2×Padding) / Stride + 1

| Input | Kernel | Padding | Stride | 출력 크기 |
|-------|--------|---------|--------|----------|
| 28 | 3 | 0 | 1 | ? |
| 28 | 3 | 1 | 1 | ? |
| 28 | 5 | 2 | 1 | ? |
| 32 | 3 | 0 | 2 | ? |
| 224 | 7 | 3 | 2 | ? |

### 문제 3-3: MaxPooling
입력 (4×4):
```
1 3 2 0
0 5 1 2
3 1 0 4
2 0 3 1
```
2×2 MaxPool, stride=2의 결과는?

---

## Level 4: RNN

### 문제 4-1: Hidden State 계산
W_xh=[[0.1, 0.2], [0.3, 0.4]], W_hh=[[0.5, 0.1], [0.2, 0.3]], b_h=[0, 0]
h_0=[0, 0]
입력 시퀀스: x=[[1, 0], [0, 1], [1, 1]]

3 timestep의 hidden state를 계산하세요:
- h_1 = tanh(x_0 @ W_xh + h_0 @ W_hh + b_h) = ?
- h_2 = tanh(x_1 @ W_xh + h_1 @ W_hh + b_h) = ?
- h_3 = tanh(x_2 @ W_xh + h_2 @ W_hh + b_h) = ?

### 문제 4-2: Vanishing Gradient
tanh 미분의 최대값은 1 (x=0), 평균은 약 0.65.
gradient가 50 timestep 역전파되면 크기가 얼마나 줄어드는가?
→ 0.65^50 = ?

---

## Level 5: LSTM

### 문제 5-1: Gate 계산
input_size=2, hidden_size=2
x_t=[1, 0.5], h_prev=[0, 0], c_prev=[0, 0]
W_f=[[0.1, 0.2, 0.3, 0.4], [0.2, 0.1, 0.4, 0.3]] (concat [x, h]용, 크기 4→2)
b_f=[0, 0]
(W_i, W_c, W_o도 같은 크기, 값은 임의)

concat = [x_t, h_prev] = [1, 0.5, 0, 0]
f_t = sigmoid(concat @ W_f + b_f) = ?

나머지 gate도 계산하세요.

---

## Level 6: Attention & Transformer

### 문제 6-1: Dot-Product Attention
Encoder hidden states:
- h1 = [0.5, 0.1]
- h2 = [0.2, 0.8]
- h3 = [0.4, 0.3]

Decoder hidden: d = [0.3, 0.7]

1. Score = d · h_i (내적) 각각 계산
2. Weights = softmax(scores) 계산
3. Context = Σ(weight_i × h_i) 계산

### 문제 6-2: Self-Attention
X = [[1, 0], [0, 1], [1, 1]] (3 토큰, 2-dim)
W_Q = W_K = W_V = I (단위행렬)

1. Q, K, V 계산
2. Scores = Q @ K^T / √d_k 계산
3. Weights = softmax(Scores) 계산
4. Output = Weights @ V 계산

### 문제 6-3: Transformer Block 그리기
아무것도 보지 말고, Transformer Encoder Block의 전체 구조를 그리세요.
(Input부터 Output까지, 모든 component 포함)
