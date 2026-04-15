# Phase 2: 신경망으로 가라 (TF2/Keras)

> **"Phase 1에서 sklearn으로 했던 것을 신경망으로 다시 한다"**

---

## 이 Phase의 목표

TF2/Keras로 **#0→#1→#2→#3→#4 코드 구조**를 처음 익히고,
같은 데이터셋에 회귀/이진분류/다중분류를 적용한다.

Phase 2가 끝나면 **"아무 데이터셋 줘도 TF2로 돌릴 수 있다"**.

## 구성

### 이론 (2-1 ~ 2-2)

| 번호 | 주제 | 핵심 질문 |
|------|------|---------|
| 2-1 | 뉴런과 신경망 구조 | "Dense(1)이 선형회귀와 같다고?" |
| 2-2 | 활성화 함수 | "왜 ReLU를 쓰는가?" |

### 실습 (2-3 ~ 2-8)

| 번호 | 주제 | 변화 포인트 |
|------|------|-----------|
| 2-3 | TF2 첫 코드 (회귀) | #0→#1→#2→#3→#4 뼈대 자체 |
| 2-4 | 회귀 — 실제 데이터셋 | #1만 바뀜 |
| 2-5 | 이진분류 (Sigmoid + BCE) | #3에 sigmoid + loss 변경 |
| 2-6 | 다중분류 (Softmax + CCE) | #3에 softmax, #2에 OneHot |
| 2-7 | 5종 데이터셋 전부 적용 | 반복 연습 (혼자 짜보기) |
| 2-8 | validation + EarlyStopping | #3에 콜백 추가 |

## Phase 1 → Phase 2 연결

| Phase 1에서 배운 것 | Phase 2에서 코드로 나오는 곳 |
|-------------------|-------------------------|
| 1-5 StandardScaler | #2 scaler.fit_transform() |
| 1-7 선형회귀 (y=wx+b) | 2-3 Dense(1) = 같은 구조 |
| 1-8 MSE | 2-3 loss='mse' |
| 1-9 경사하강법 | 2-3 optimizer='adam' |
| 1-10 Sigmoid | 2-5 activation='sigmoid' |
| 1-10 Softmax | 2-6 activation='softmax' |
| 1-11 BCE | 2-5 loss='binary_crossentropy' |
| 1-11 CCE | 2-6 loss='categorical_crossentropy' |
| 1-13 과적합 | 2-8 EarlyStopping |

## 레퍼런스

- Coursera C2W1~W2 Labs (Neurons, CoffeeRoasting, ReLU, Softmax, Multiclass)
- AI5-main keras 01~25
- Google MLCC "Neural Networks"

## 보고 오기 (Phase 2 전체)

- 3Blue1Brown "Neural Networks" Ch.1~2 (필수)
- 3Blue1Brown "Backpropagation" Ch.3~4 (필수)
- 구글 검색: "활성화 함수 종류 비교"
