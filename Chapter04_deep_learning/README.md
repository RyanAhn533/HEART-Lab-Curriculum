# Chapter 4: 신경망으로 가라 (TF2/Keras)

> **"Chapter 2에서 sklearn으로 했던 것을 신경망으로 다시 한다"**

---

## 이 챕터의 목표

TF2/Keras로 **#0→#1→#2→#3→#4 코드 구조**를 처음 익히고,
같은 데이터셋에 회귀/이진분류/다중분류를 적용한다.

Chapter 4가 끝나면 **"아무 데이터셋 줘도 TF2로 돌릴 수 있다"**.

## 구성

### 이론 (4-1 ~ 4-2)

| 번호 | 주제 | 핵심 질문 |
|------|------|---------|
| 4-1 | 뉴런과 신경망 구조 | "Dense(1)이 선형회귀와 같다고?" |
| 4-2 | 활성화 함수 | "왜 ReLU를 쓰는가?" |

### 실습 (4-3 ~ 4-8)

| 번호 | 주제 | 변화 포인트 |
|------|------|-----------|
| 4-3 | TF2 첫 코드 (회귀) | #0→#1→#2→#3→#4 뼈대 자체 |
| 4-4 | 회귀 — 실제 데이터셋 | #1만 바뀜 |
| 4-5 | 이진분류 (Sigmoid + BCE) | #3에 sigmoid + loss 변경 |
| 4-6 | 다중분류 (Softmax + CCE) | #3에 softmax, #2에 OneHot |
| 4-7 | 5종 데이터셋 전부 적용 | 반복 연습 — `4-7_혼자짜기_5종데이터셋.md` (혼자 짜보기) |
| 4-8 | validation + EarlyStopping | #3에 콜백 추가 |

## Chapter 2 → Chapter 4 연결

| Chapter 2에서 배운 것 | Chapter 4에서 코드로 나오는 곳 |
|-------------------|-------------------------|
| 1-6 StandardScaler | #2 scaler.fit_transform() |
| 2-2 선형회귀 (y=wx+b) | 4-3 Dense(1) = 같은 구조 |
| 2-3 MSE | 4-3 loss='mse' |
| 2-4 경사하강법 | 4-3 optimizer='adam' |
| 2-5 Sigmoid | 4-5 activation='sigmoid' |
| 2-5 Softmax | 4-6 activation='softmax' |
| 2-6 BCE | 4-5 loss='binary_crossentropy' |
| 2-6 CCE | 4-6 loss='categorical_crossentropy' |
| 2-8 과적합 | 4-8 EarlyStopping |

## 레퍼런스

- Coursera C2W1~W2 Labs (Neurons, CoffeeRoasting, ReLU, Softmax, Multiclass)
- AI5-main keras 01~25
- Google MLCC "Neural Networks"

## 보고 오기 (Chapter 4 전체)

- 3Blue1Brown "Neural Networks" Ch.1~2 (필수)
- 3Blue1Brown "Backpropagation" Ch.3~4 (필수)
- 구글 검색: "활성화 함수 종류 비교"
