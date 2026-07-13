# HEART Lab AI Curriculum — Master Document

---

## 1. 교육 이념

### 한 줄 정의

> **"AI로 문제를 풀 수 있는 사람을 만든다"**

### 왜 이렇게 가르치는가

**시장 현실:**
- AI 경력직 요구 비율: 54% → 80.6%
- 2026년 신입 채용 비율: 12.4%
- SW 신입 비중 2년 변화: 53.5% → 37.4%
- 기업은 교육시켜도 1~2년 만에 이직 → **바로 투입 가능한 사람만 원한다**

**바로 투입이란:**
- "YOLO 써봤습니다" X
- "이 문제에 왜 YOLO인지 설명하고, 설계하고, 구현할 수 있습니다" O
- 코딩은 기본, **뭘 만들지 아는 것까지** 되어야 함

**Top-Down Thinking:**
- First Principles — 근본으로 돌아가서 문제를 재정의
- 기존 DMS: 졸음 감지 → HEART Lab: 감정 변화의 근본 원인 추적
- 교육도 Top-Down: **일단 돌려보고 → 왜 그런지 파고든다**
- 수학부터 X, 결과부터 보고 필요한 이론을 역추적

### 3단계 성장 모델

| 단계 | 목표 | 기준 |
|------|------|------|
| 1단계: 문제 해결 사고 | "이 문제 어떻게 풀래?" 바로 떠오를 때까지 반복 | Phase 2~3 |
| 2단계: 모델을 도구로 | "이 문제에 왜 이 모델인지" 설명할 수 있는 것 | Phase 4~8 |
| 3단계: 서비스 구현 | 챗봇, 실시간 탐지 — 혼자 만들 수 있는 것 | Phase 9~10 |

---

## 2. 교육 원칙

### 코드 5대 원칙

| 원칙 | 설명 |
|------|------|
| **구조 통일** | 모든 코드가 `#0→#1→#2→#3→#4` 동일 뼈대 |
| **하나만 바뀐다** | 이전 코드에서 변경된 건 항상 1개 섹션 |
| **같은 데이터, 다른 기법** | 5개 기본 데이터셋을 끝까지 반복 적용 |
| **간결하게** | TF2/Keras의 간결한 API만, 복잡한 건 안 씀 |
| **60~120줄** | 파일당 최대 120줄, 그 이상이면 분리 |

### 코드 뼈대 (모든 파일 공통)

```python
# ============================================
# [번호] [주제] — [데이터셋]
# 이전과 차이: [한 줄로 뭐가 바뀌었는지]
#
# 왜 배우는가:
#   [이 기법이 왜 필요한지, HEART Lab 과제와의 연결]
#
# ▶ 보고 오기: [필수 영상/블로그]
# ▶ 나중에 읽기: [논문]
# ============================================

#0. 라이브러리 ──────────────────────────────

#1. 데이터 ─────────────────────────────────

#2. 데이터 전처리 및 분할 ───────────────────

#3. 모델 ──────────────────────────────────

#4. 평가 ──────────────────────────────────
```

**규칙:**
- 헤더에 "이전과 차이" 반드시 명시
- 섹션 구분선으로 시각적 분리
- 주석은 최소한 — 코드가 말하게
- 새로 등장하는 줄만 `# ← NEW` 표시

### 기술 스택 규칙

| 구간 | 프레임워크 | 이유 |
|------|-----------|------|
| Phase 0~8 | **TensorFlow2/Keras만** | 간결, 선언적, 결과 빨리 봄 |
| Phase 9 | **TF2 + sklearn** | ML 기법은 sklearn이 표준 |
| Phase 10 | **PyTorch** | 내부 동작 이해, 논문 재현 |

### TF2 — 쓰는 것 / 안 쓰는 것

**쓰는 것:**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, LSTM, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```

**안 쓰는 것:**
- `tf.GradientTape` → 복잡, Torch에서 배울 내용
- `tf.data.Dataset` → 복잡
- `tf.function` → 복잡
- Custom Training Loop → Torch에서
- Functional API → 후반부에 간단히만

### 5개 기본 데이터셋

| 데이터셋 | 유형 | 특성 수 | 용도 |
|---------|------|--------|------|
| Boston Housing | 회귀 | 13 | 기본 회귀 |
| Diabetes | 회귀 | 10 | 의료 데이터 |
| Iris | 다중분류 | 4 | 분류 입문 |
| Breast Cancer | 이진분류 | 30 | 이진분류 |
| MNIST | 이미지분류 | 28x28 | CNN 이후 |

매 새로운 기법마다 해당 유형의 데이터셋 전부에 적용.

---

## 3. 이론 학습 원칙

### 순서

```
영상 (직관) → 블로그 리뷰 (정리) → 논문 (원본)
절대 논문부터 읽지 마라.
```

### 리소스 우선순위

**1순위: 3Blue1Brown (영상)**
- 시각적 직관 최강. 한글자막 있음.
- 이해 안 되면 3Blue1Brown부터 찾아라.
- 핵심 시리즈:
  - Essence of Linear Algebra (선형대수)
  - Neural Networks (뉴럴넷)
  - Essence of Calculus (미적분)
  - Transformers (트랜스포머)

**2순위: 구글 검색 "[주제] 논문 리뷰" (블로그)**
- "LSTM 논문 리뷰", "ResNet 논문 리뷰" 이렇게 검색
- 한국어 블로그 2~3개 읽으면 핵심 파악 가능
- 논문 원문 읽기 전에 리뷰 먼저 보면 맥락이 잡힘

**3순위: 특정 추천 블로그/채널**
- Transformer 계열: https://codingopera.tistory.com/43
- StatQuest (통계/ML 기초, 영어)
- 혁펜하임 (한국어 DL 전반)

---

## 4. 운영 방식

### 단일 트랙

```
학부/석사 구분 없음. 전부 같은 트랙.
아는 사람은 스킵. 모르면 처음부터.
```

### 진입 방식

| 진단 결과 | 시작 지점 |
|----------|----------|
| 전부 모름 | Phase 0부터 |
| Phase 2까지 알면 | Phase 3부터 |
| Phase 6까지 알면 | Phase 7부터 |
| 전부 알면 | 논문 재현 프로젝트만 |

**스킵 기준:** 해당 Phase 코드를 보고 **30분 안에 혼자 짤 수 있으면** 넘어감. 못 짜면 처음부터.

### 평가

| 시점 | 평가 | 범위 |
|------|------|------|
| Phase 3 끝 | 체크포인트 시험 1 | 기초 회귀/분류 체화 |
| Phase 6 끝 | 체크포인트 시험 2 | CNN/RNN 체화 |
| Phase 10 끝 | 최종 발표 | 논문 재현 프로젝트 |
| 매월 | 월간 발표 | 학습 현황 + 결과물 |

### 속도

진도는 사람마다 다름.

| 유형 | 예상 기간 |
|------|----------|
| 빠른 사람 | 8주 |
| 보통 | 12~16주 |
| 느린 사람 | 20주 |

---

## 5. 전체 로드맵

### Phase 0: 왜 하는가

| 번호 | 주제 | 핵심 메시지 |
|------|------|-----------|
| 0-1 | AI 취업시장 현실 | 경력직 80%, 신입 12.4% |
| 0-2 | 바로 투입 = 문제 해결 능력 | 코딩 + 뭘 만들지 아는 것 |
| 0-3 | 세계 트렌드 | NVIDIA, Affectiva, LLM, 멀티모달, Physical AI |
| 0-4 | HEART Lab 과제 | 현대차 감정인식, 두산 협동로봇, 토론토대 국제공동 |
| 0-5 | Top-Down Thinking | First Principles, 근본으로 돌아가서 재정의 |
| 0-6 | 커리큘럼 소개 | #0→#1→#2→#3→#4 구조, 3단계 성장 모델 |

> 보고 오기: 3Blue1Brown "But what is a neural network?"

### Phase 1: 통계/수학 기초 (코드에서 만날 개념의 뿌리)

| 번호 | 주제 | 나중에 만나는 곳 |
|------|------|---------------|
| 1-1 | 데이터를 보는 눈 (평균, 분산, 표준편차, 정규분포) | `StandardScaler()` |
| 1-2 | 상관관계 (상관계수, scatter plot) | `Feature Importance` |
| 1-3 | 확률 (조건부확률, Bayes) | `Softmax 출력` |
| 1-4 | 손실함수 직관 (MSE, Cross-Entropy) | `model.compile(loss=...)` |
| 1-5 | 경사하강법 직관 (산에서 내려가기) | `optimizer='adam'` |

> 보고 오기: 3Blue1Brown "Gradient descent", 구글 검색 "경사하강법 직관"

**Phase 1은 수학 수업이 아니다.** 코드에서 만날 개념을 미리 직관적으로 이해시키는 것. 수식 전개 X, 그래프와 예시로 "아 이런 거구나" 수준.

> ★ 단, 딥러닝 진입(Phase 2/Chapter 4) 직전에 **손계산 부트캠프**(`Chapter03_5_math_bootcamp/`)를 통과할 것.
> 직관(영상)만으로 역전파를 넘으려던 학생이 미끄러지는 지점을 메우는 다리다. 진입 진단은 DIAGNOSTIC_TEST Section A.

### Phase 2: 돌려봐 (TF2/Keras)

| 번호 | 주제 | AI5 대응 | 변화 포인트 |
|------|------|---------|-----------|
| 02 | 기본 회귀 (Sequential + Dense) | keras01-04 | #0~#4 뼈대 자체 |
| 03 | 다중입력 + 깊은 네트워크 | keras05-07 | #3에 레이어 추가 |
| 04 | train/test 분할 + R2 | keras08-10 | #2에 split 추가 |
| 05 | 실제 데이터셋 5종 적용 | keras11-13 | #1에 실제 데이터 |

> 보고 오기: 3Blue1Brown "Neural Networks" Ch.1, 구글 검색 "퍼셉트론 논문 리뷰"

### Phase 3: 왜 안 되지? (TF2/Keras)

| 번호 | 주제 | AI5 대응 | 변화 포인트 |
|------|------|---------|-----------|
| 06 | validation + EarlyStopping | keras14-19 | #3에 콜백 추가 |
| 07 | 이진분류 (Sigmoid + BCE) | keras20-21 | #3에 sigmoid, loss 변경 |
| 08 | 다중분류 (Softmax + OneHot) | keras22-25 | #3에 softmax, #2에 인코딩 |
| 09 | Scaler (MinMax, Standard) | keras26-27 | #2에 Scaler 추가 |

> 보고 오기: 3Blue1Brown "Backpropagation" Ch.3~4, 구글 검색 "과적합 해결 방법"

**★ 체크포인트 시험 1**

### Phase 4: 저장하고 구조 바꾸기 (TF2/Keras)

| 번호 | 주제 | AI5 대응 | 변화 포인트 |
|------|------|---------|-----------|
| 10 | 모델 저장/로드 + Checkpoint | keras28-32 | #3 뒤에 save 추가 |
| 11 | Dropout | keras32 | #3에 Dropout 레이어 |
| 12 | 함수형 API (간단히) | keras33-34 | #3 구조 변경 |

### Phase 5: 이미지 (TF2/Keras)

| 번호 | 주제 | AI5 대응 | 변화 포인트 |
|------|------|---------|-----------|
| 13 | CNN 기초 (Conv2D + MaxPool) | keras35-37 | #1 이미지, #2 reshape, #3 Conv2D |
| 14 | DNN vs CNN 비교 | keras38-39 | #3만 교체해서 비교 |
| 15 | 데이터 증강 | keras40-41 | #2에 ImageDataGenerator |

> 보고 오기: 3Blue1Brown "But what is a convolution?", 구글 검색 "AlexNet 논문 리뷰"

### Phase 6: 시계열 (TF2/Keras)

| 번호 | 주제 | AI5 대응 | 변화 포인트 |
|------|------|---------|-----------|
| 16 | SimpleRNN | keras51-52 | #2에 3D reshape, #3에 RNN |
| 17 | LSTM | keras53-54 | #3에 RNN → LSTM 교체 |
| 18 | Bidirectional + Conv1D | keras56-58 | #3에 Bidirectional 래핑 |

> 보고 오기: 3Blue1Brown "Recurrent Neural Networks", 구글 검색 "LSTM 논문 리뷰"

**★ 체크포인트 시험 2**

### Phase 7: NLP (TF2/Keras)

| 번호 | 주제 | AI5 대응 | 변화 포인트 |
|------|------|---------|-----------|
| 19 | Tokenizer + Embedding | keras64-66 | #1 텍스트, #2 토큰화, #3 Embedding |
| 20 | 감정분류 프로젝트 | — | 종합 적용 |

> 보고 오기: 3Blue1Brown "Word2Vec", 구글 검색 "Word2Vec 논문 리뷰"

### Phase 8: 성능 올리기 (TF2 + sklearn)

| 번호 | 주제 | AI5 대응 | 변화 포인트 |
|------|------|---------|-----------|
| 21 | 옵티마이저 + ReduceLROnPlateau | keras68-69 | #3에 콜백 추가 |
| 22 | 하이퍼파라미터 튜닝 | keras71-72 | #3을 SearchCV로 래핑 |
| 23 | 전이학습 (VGG16, ResNet, GAP) | keras74-79 | #3에 사전학습 모델 |

> 보고 오기: 구글 검색 "VGGNet 논문 리뷰", "ResNet 논문 리뷰"

### Phase 9: ML 기본기 (sklearn)

| 번호 | 주제 | AI5 대응 | 변화 포인트 |
|------|------|---------|-----------|
| 24 | PCA + KFold | m01-10 | #2에 PCA, #4에 cross_val |
| 25 | GridSearch + Feature Importance | m14-29 | #3을 GridSearchCV로 래핑 |
| 26 | 앙상블 (Bagging, Voting, Stacking) | m37-40 | #3에 앙상블 모델 |
| 27 | SMOTE, PolynomialFeatures | m19-20, m41-44 | #2에 SMOTE/Poly 추가 |

> 보고 오기: 3Blue1Brown "Essence of Linear Algebra" 전편, 구글 검색 "XGBoost 논문 리뷰"

### Phase 10: PyTorch

| 번호 | 주제 | AI5 대응 | 변화 포인트 |
|------|------|---------|-----------|
| 28 | Keras에서 한 것 Torch로 재구현 | torch01-10 | 같은 #1~#4, 다른 프레임워크 |
| 29 | DataLoader, nn.Module | torch11-12 | #1에 DataLoader |
| 30 | CNN + RNN in Torch | torch13-22 | #3에 Conv2d, LSTM |
| 31 | 논문 재현 프로젝트 | — | Transformer, Attention |

> 보고 오기: 3Blue1Brown "Attention in transformers", codingopera.tistory.com/43
> 필수 논문: Vaswani et al. 2017 "Attention Is All You Need"

**★ 최종 발표 → 과제 투입**

---

## 6. "하나만 바뀐다" — 변화 포인트 추적

### Phase 2: 뼈대 익히기

```
02 기본 회귀        → #0~#4 뼈대 자체를 처음 익힘
03 다중입력/깊은 net → #3에 레이어 추가 (Dense 1개 → 5개)
04 train/test       → #2에 train_test_split 추가
05 실제 데이터셋     → #1에 load_boston 등 (나머지 동일)
```

### Phase 3: 각 섹션에 1개씩 추가

```
06 EarlyStopping    → #3에 validation_split + EarlyStopping 추가
07 이진분류         → #3에 sigmoid, #0에 binary_crossentropy      ← NEW
08 다중분류         → #3에 softmax, #2에 to_categorical           ← NEW
09 Scaler          → #2에 StandardScaler 추가                    ← NEW
```

### Phase 4: #3 구조 변경

```
10 저장/로드        → #3 뒤에 model.save() 추가
11 Dropout         → #3에 Dropout 레이어 추가
12 함수형 API      → #3을 Sequential → Functional API로 교체
```

### Phase 5: #1, #2 변경 (이미지)

```
13 CNN             → #1에 MNIST, #2에 reshape(28,28,1), #3에 Conv2D
14 DNN vs CNN      → #3만 DNN vs CNN 교체해서 비교
15 증강            → #2에 ImageDataGenerator 추가
```

### Phase 6: #2 변경 (시계열)

```
16 RNN             → #2에 reshape(timesteps, features), #3에 SimpleRNN
17 LSTM            → #3에 SimpleRNN → LSTM 교체
18 Bi + Conv1D     → #3에 Bidirectional() 래핑 + Conv1D
```

### Phase 7: #1, #2 변경 (텍스트)

```
19 NLP             → #1에 텍스트, #2에 Tokenizer+pad_sequences, #3에 Embedding
20 프로젝트         → 전체 종합 적용
```

### Phase 8: #3 최적화

```
21 Optimizer       → #3에 ReduceLROnPlateau 추가
22 튜닝            → #3을 RandomSearchCV로 래핑
23 전이학습        → #3에 VGG16(include_top=False) + trainable=False
```

### Phase 9: sklearn

```
24 PCA+KFold       → #2에 PCA 추가, #3에 RandomForest
25 GridSearch      → #3을 GridSearchCV로 래핑
26 앙상블          → #3에 VotingClassifier, StackingClassifier
27 SMOTE+Poly      → #2에 SMOTE/PolynomialFeatures 추가
```

### Phase 10: PyTorch

```
28 Torch 재구현    → 같은 #1~#4인데 nn.Module + 수동 train loop
29 DataLoader      → #1에 TensorDataset + DataLoader
30 CNN+RNN Torch   → #3에 Conv2d, nn.LSTM
31 논문 재현       → Transformer 구현
```

---

## 7. 이론 매핑 전체표

| Phase | 코드에서 만나는 것 | 보고 오기 (필수) | 나중에 읽기 (선택) |
|-------|-----------------|----------------|-----------------|
| **0** | — | 3B1B "Neural Network" | — |
| **1** | Scaler, Loss, Optimizer | 3B1B "Gradient descent" | — |
| **2** | Sequential, Dense | 3B1B "Neural Networks" Ch.1 | 퍼셉트론 논문 리뷰 |
| **3** | EarlyStopping, Sigmoid, Softmax | 3B1B "Backpropagation" | 과적합 블로그 |
| **4** | Dropout, save/load | StatQuest "Dropout" | Dropout 논문 (2014) |
| **5** | Conv2D, MaxPool | 3B1B "Convolution" | AlexNet 논문 리뷰 |
| **6** | RNN, LSTM | 3B1B "RNN" | LSTM 논문 리뷰 |
| **7** | Embedding, Tokenizer | 3B1B "Word2Vec" | Word2Vec 논문 리뷰 |
| **8** | VGG16, ResNet, Transfer | 구글 "VGGNet 논문 리뷰" | VGG/ResNet 논문 |
| **9** | PCA, KFold, XGBoost | 3B1B "Linear Algebra" 전편 | XGBoost 논문 리뷰 |
| **10** | Transformer, Attention | 3B1B "Transformers" | Attention Is All You Need |
| | | codingopera.tistory.com/43 | |

### 핵심 채널 정리

| 채널 | 언어 | 특징 | 용도 |
|------|------|------|------|
| **3Blue1Brown** | 영어(한글자막) | 시각적 직관 최강 | 전 Phase, 최우선 |
| **혁펜하임** | 한국어 | 짧고 명확 | DL 전반 |
| **StatQuest** | 영어 | 통계+ML 기초 | Phase 1, 9 |
| **codingopera** | 한국어 | Transformer 정리 | Phase 10 |

### 필수 논문 (읽는 시점)

| 시점 | 논문 | 왜 읽는가 |
|------|------|---------|
| Phase 5 후 | LeNet-5 (LeCun, 1998) | CNN의 시작 |
| Phase 5 후 | AlexNet (Krizhevsky, 2012) | 딥러닝 부활 |
| Phase 6 후 | LSTM (Hochreiter & Schmidhuber, 1997) | 시계열의 핵심 |
| Phase 7 후 | Word2Vec (Mikolov, 2013) | 임베딩의 시작 |
| Phase 8 | VGGNet (Simonyan, 2014) | 전이학습 기본 |
| Phase 8 | ResNet (He, 2015) | Skip Connection |
| Phase 10 | Attention Is All You Need (Vaswani, 2017) | 현대 AI의 기반 |

### 이론 공부하는 법 (학생 가이드)

1. **영상부터**: 3Blue1Brown에서 해당 주제 검색
   - 선형대수 → "Essence of Linear Algebra" 시리즈
   - 뉴럴넷 → "Neural Networks" 시리즈
   - 없으면 혁펜하임, StatQuest 검색

2. **블로그 리뷰**: 구글에 "[주제] 논문 리뷰" 검색
   - 예: "ResNet 논문 리뷰", "Word2Vec 논문 리뷰"
   - 한국어 블로그 2~3개 읽으면 핵심 파악
   - 추천: codingopera.tistory.com (Transformer 계열)

3. **논문 원문**: 리뷰 읽고 나서, 여유 있을 때
   - 처음부터 논문 읽으면 막힘
   - 리뷰로 맥락 잡고 → 원문에서 디테일 확인

**순서: 영상(직관) → 블로그 리뷰(정리) → 논문(원본). 절대 논문부터 읽지 마라.**

---

## 8. 세계 트렌드와 HEART Lab 연결

### 글로벌 트렌드

| 분야 | 키워드 | 대표 |
|------|--------|------|
| 감정 AI | Emotion AI, Affective Computing | NVIDIA Audio2Face, Affectiva (Smart Eye) |
| LLM | 대규모 언어모델 | GPT, Claude, Gemini |
| AI Agent | 자율적 의사결정 | 자율주행, 로봇 제어 |
| 멀티모달 | 영상 + 음성 + 텍스트 | GPT-4V, Gemini |
| Physical AI | 로봇, 자율주행 | NVIDIA Isaac, Figure AI |

### HEART Lab 과제

| 과제 | 파트너 | 내용 | 커리큘럼 연결 |
|------|--------|------|-------------|
| 차량 감정인식 | 현대자동차 | 멀티모달 센싱 → 운전자 감정 → 안전운전 | Phase 5~7 (CNN, RNN, NLP) |
| 협동로봇 AI | 두산 | VLM/VLA + AI SoC + 강화학습 | Phase 8~10 (전이학습, Torch) |
| 디지털 트윈 | 토론토대 | EV 배터리 + AI Agent 제조 자동화 | Phase 10 (논문 재현) |

---

## 9. 폴더 구조

```
HEART-Lab-Curriculum/
├── README.md                  ← 전체 소개
├── CURRICULUM_MASTER.md       ← 이 문서 (전체 설계)
├── DIAGNOSTIC_TEST.md         ← 진단 테스트 40문제
├── MANUAL.md                  ← 교육 운영 매뉴얼
├── ROADMAP.md                 ← 학생용 전체 지도
├── PAPER_GUIDE.md             ← 논문 7편 읽기 가이드
├── Chapter03_5_math_bootcamp/ ← ★ 수학 손계산 부트캠프 (Ch.3→4 다리)
├── worksheets/handcalc_*.md   ← CNN/RNN/LSTM 손계산 워크시트
│
├── Phase00_why/               ← 왜 하는가 (문서 + PPT)
├── Phase01_foundations/       ← 통계/수학 기초 (문서 + 시각화 코드)
├── Phase02_get_started/       ← 돌려봐
├── Phase03_debugging/         ← 왜 안 되지?
├── Phase04_save_structure/    ← 저장, 구조
├── Phase05_image/             ← CNN
├── Phase06_timeseries/        ← RNN/LSTM
├── Phase07_nlp/               ← NLP
├── Phase08_optimization/      ← 성능 올리기
├── Phase09_ml/                ← ML 기본기
├── Phase10_pytorch/           ← PyTorch + 논문 재현
│
├── worksheets/
│   ├── checkpoint_1.md        ← Phase 3 끝
│   ├── checkpoint_2.md        ← Phase 6 끝
│   └── final_project.md      ← 최종 프로젝트
│
└── AI5-main/                  ← 원본 학습 코드 (참고용)
    ├── keras/
    ├── keras2/
    ├── ml/
    └── torch/
```

---

## 10. 한 문장 요약

> **#0→#1→#2→#3→#4 뼈대를 몸에 익히고, 한 번에 하나씩만 바꿔가며, 같은 데이터에 다른 기법을 적용하면서, 3Blue1Brown과 논문 리뷰로 "왜"를 이해하고, HEART Lab의 실제 과제에 투입될 수 있는 연구원을 만든다.**
