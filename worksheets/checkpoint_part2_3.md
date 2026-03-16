# Part 2~3 체크포인트

자기 점검표. 각 항목을 **안 보고** 할 수 있어야 통과.

## Part 2: 학습의 원리

### 2-1. 선형회귀
- [ ] "AI = 오차를 줄이는 과정"을 자기 말로 설명할 수 있다
- [ ] y=wx+b에서 Forward → Loss(MSE) → Gradient → Update 과정을 설명할 수 있다
- [ ] w=0, b=0, lr=0.01에서 **10 epoch 손계산** 결과가 numpy와 소수점 4자리까지 일치
- [ ] Learning Rate가 너무 크면/작으면 어떻게 되는지 설명할 수 있다

### 2-2. ANN (MLP)
- [ ] 퍼셉트론 하나로 AND는 되고 XOR은 안 되는 이유를 설명할 수 있다
- [ ] MLP 2층의 Forward Pass를 종이에 계산할 수 있다
- [ ] **Chain Rule**을 이용한 역전파를 종이에 전개할 수 있다 (안 보고!)
- [ ] numpy, Keras, PyTorch 각각으로 MLP를 구현할 수 있다
- [ ] "프레임워크가 해주는 것 = 내가 손으로 한 것과 같은 수학"임을 이해했다

## Part 3: 데이터 유형별 모델

### 3-1. CNN
- [ ] MLP로 이미지를 처리하면 안 되는 이유를 설명할 수 있다
- [ ] Convolution 연산을 손으로 계산할 수 있다 (4×4 입력, 3×3 필터)
- [ ] 출력 크기 공식: (Input - Kernel + 2×Padding) / Stride + 1 을 5문제 연속 맞출 수 있다
- [ ] MaxPooling의 동작을 설명하고 손으로 계산할 수 있다
- [ ] CNN 파이프라인 (Conv→ReLU→Pool→...→Dense) 을 그릴 수 있다

### 3-2. RNN
- [ ] "순서가 중요한 데이터"에 CNN/MLP가 부족한 이유를 설명할 수 있다
- [ ] h_t = tanh(W_xh·x_t + W_hh·h_(t-1) + b) 를 3 timestep 손으로 계산할 수 있다
- [ ] **Vanishing Gradient**가 왜 생기는지 후배한테 수식 없이 설명할 수 있다

### 3-3. LSTM
- [ ] RNN의 Vanishing Gradient를 LSTM이 어떻게 해결하는지 설명할 수 있다
- [ ] Forget / Input / Output Gate의 역할을 각각 설명할 수 있다
- [ ] Cell State가 "고속도로"인 이유를 설명할 수 있다
- [ ] LSTM 1 timestep의 gate 출력 → cell update → hidden 계산을 손으로 할 수 있다

## 손코딩 시험

| # | 문제 | 목표 시간 | 통과 |
|---|------|----------|------|
| 1 | Linear Regression forward + backward 1 epoch | 5분 | [ ] |
| 2 | MLP 2층 Chain Rule 전개 | 10분 | [ ] |
| 3 | CNN 출력 크기 계산 5문제 | 전부 정답 | [ ] |
| 4 | RNN hidden state 3 timestep | 10분 | [ ] |
| 5 | LSTM gate 1 timestep | 15분 | [ ] |

## 코딩 목표 수치

| 태스크 | 목표 | 달성 |
|--------|------|------|
| MNIST MLP accuracy | ≥ 97% | [ ] ___% |
| CIFAR-10 CNN accuracy | ≥ 75% | [ ] ___% |
| sin 함수 LSTM MSE | ≤ 0.01 | [ ] ___ |

## "이 모델은 언제 쓰는 건데?" (자기 말로 작성)

| 모델 | 어떤 문제에 | 어떤 데이터에 | 왜 이 모델인지 |
|------|-----------|-------------|--------------|
| MLP | | | |
| CNN | | | |
| RNN | | | |
| LSTM | | | |
