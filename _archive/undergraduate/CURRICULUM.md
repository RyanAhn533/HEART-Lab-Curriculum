# HEART Lab 신입생 AI 교육 커리큘럼 (v3)

---

# Part 0: 왜 여기 있고, 뭘 하게 되는가

신입생이 제일 먼저 알아야 할 건 코드가 아니라 **맥락**이다.

## 0-1. 연구실은 뭘 하는 곳인가

- HEART Lab (세종대학교)
- 멀티모달 인공지능 연구: 카메라, 음성, 생체신호 등 여러 데이터를 결합해서 사람의 상태를 인식
- 현재 핵심 과제: **산업부 차량 과제** (RS-2024-00487049)
  - 차량 내에서 카메라(얼굴/제스처) + 음성 + 생체신호 → 운전자 감정 판단
  - 이게 너희가 투입될 과제다

## 0-2. 돈이 어떻게 도는가 — 국책과제 구조

```
정부(산업부/과기부)가 과제 공모
    ↓
연구실이 제안서 작성 → 선정
    ↓
연구비 수주 (장비, 인건비, 학회 출장 등)
    ↓
매년 연차보고서 제출 → 평가
    ↓
통과하면 다음 년도 계속 / 못하면 과제 중단
```

- 연구실 운영 = 과제 수주에 달려 있음
- 학생들이 실제로 모델 개발, 데이터 수집, 실험, 논문을 함
- **너희가 배우는 모든 기술은 이 과제에 바로 쓰이는 것들**

## 0-3. AI 하면 어디로 가는가

| 방향 | 예시 |
|------|------|
| 기업 연구소 | 삼성 리서치, LG AI연구원, 네이버, 카카오, 현대 |
| AI 스타트업 | 의료AI, 자율주행, 로봇, 생성AI |
| 연구소/대학원 | KAIST, ETRI, 한국전자통신연구원 |
| 해외 | Google, Meta, NVIDIA, OpenAI |

공통 요구 역량: **논문 읽고 구현하는 능력**, 데이터 파이프라인, 실험 설계

## 0-4. 공부를 어떻게 해야 하는가

- **코드만 돌리면 실력이 안 늘어남** — 돌아가는 원리를 수학적으로 이해해야 함
- 손계산 → numpy → 프레임워크 순서로 해야 면접에서도 설명할 수 있음
- 논문 읽는 습관: 처음엔 한글 리뷰 → 점차 원문
- 모르면 바로 물어보기, 혼자 끙끙대다 시간 날리지 말기
- 매일 조금씩 꾸준히 > 몰아서 하루 12시간

---

# Part 1: AI의 전체 그림 — 먼저 숲을 보자

코드를 한 줄도 안 짠 상태에서 "AI가 뭔지" 전체 그림부터 잡는다.
세부 모델은 나중에 하나씩 파고 든다.

## 1-1. AI란 무엇인가 — 한 문장 정의

> **인공지능 = Prediction(예측)과 Ground Truth(실제값)의 오차를 줄여나가는 과정**

세상의 모든 AI 모델은 결국 이 구조:

```
Input → Model → Prediction → Loss(pred, y) → Backward(Chain Rule) → Weight Update
                                   ↑
                            "이 오차를 줄이는 게 학습"
```

ChatGPT도, 자율주행도, YOLO도, 감정인식도 — 전부 이 파이프라인이다.
모델 이름만 다르지 뼈대는 같다. 이걸 이해하면 새 모델이 나와도 겁나지 않는다.

## 1-2. AI를 잘한다는 건 — 문제 해결 사고 흐름

모델 이름 많이 아는 게 실력이 아니다.
**진짜 실력 = 문제를 보고 풀이 방법이 바로 떠오르는 것.**

```
문제가 뭐지? → 데이터를 어떻게 쓰지? → 전처리를 어떻게? → 어떤 모델? → 평가는?
```

항상 이 순서. 모델부터 고르는 게 아니라, **문제부터 정의**한다.

### Step 1: 문제가 뭐지?

제일 먼저: **출력이 뭔지** 정확히 정의

| 문제 유형 | 출력 | 예시 | Loss |
|----------|------|------|------|
| 회귀 | 숫자 (연속값) | 집값, 매출, 온도 | MSE, MAE |
| 이진 분류 | 0 or 1 | 스팸, 합격, 긍/부정 | Binary CE |
| 다중 분류 | N개 중 하나 | 감정 7종, 숫자 0~9 | Categorical CE |
| 시계열 예측 | 미래 값 | 주가, 센서 | MSE |
| 객체 탐지 | 박스 + 클래스 | 안전모, 차량 | Box + Class Loss |
| 세그멘테이션 | 픽셀마다 클래스 | 의료 영상, 도로 | Dice, CE |
| 생성 | 새 데이터 | 이미지, 텍스트, 번역 | Adversarial, NLL |

### Step 2: 데이터가 뭐지?

| 형태 | 예시 | 특징 |
|------|------|------|
| 정형 (표) | 매출, 고객, 센서 로그 | 행=샘플, 열=피처 |
| 이미지 | 얼굴, X-ray | 2D 공간 구조 |
| 텍스트 | 리뷰, 뉴스, 대화 | 가변 길이 |
| 시계열 | 주가, 날씨, ECG | 시간 순서가 핵심 |
| 영상 | CCTV, 행동 인식 | 이미지의 시퀀스 |
| 오디오 | 음성, 감정 | 파형 → spectrogram 변환 |

체크: 데이터 양은? 라벨 있는가? 클래스 불균형은?

### Step 3: 전처리를 어떻게?

| 데이터 | 필수 전처리 | 선택 전처리 |
|--------|-----------|------------|
| 정형 | 결측치, 스케일링 | PCA, 피처 선택, 로그 변환 |
| 이미지 | resize, /255 정규화 | augmentation, 히스토그램 균일화 |
| 텍스트 | 토크나이징, 패딩 | 불용어 제거, BPE |
| 시계열 | 윈도우 슬라이딩, 정규화 | 차분, 이동평균 |
| 오디오 | MFCC / mel-spectrogram | 노이즈 제거, 증강 |
| 영상 | 프레임 추출, resize | optical flow |

원칙:
- Train/Test 분리를 전처리 **전에** (데이터 누수 방지)
- 스케일링은 train에 fit → test에 transform만
- augmentation은 train에만

### Step 4: 어떤 모델?

| 상황 | 선택 | 이유 |
|------|------|------|
| 정형 + 적은 양 | XGBoost, RandomForest | 딥러닝은 데이터 많아야 유리 |
| 정형 + 많은 양 | MLP, TabNet | 딥러닝도 가능 |
| 이미지 분류 | CNN (ResNet, EfficientNet) | 공간 특징 추출에 최적 |
| 이미지 + 데이터 적음 | Transfer Learning | 처음부터 학습하면 과적합 |
| 텍스트 분류 | BERT, KoBERT | 사전학습 언어 모델 Fine-tuning |
| 시계열 | LSTM, Transformer | 순서 의존성 |
| 실시간 탐지 | YOLO | 속도 중요 |
| 정밀 탐지 | Faster R-CNN | 정확도 중요 |
| 번역/요약/생성 | Transformer (GPT, T5) | 현재 최고 성능 |

### Step 5: 잘 되는지 확인

| 문제 유형 | 평가 지표 | 주의 |
|----------|----------|------|
| 회귀 | MSE, RMSE, R² | R²는 1에 가까울수록 |
| 분류 | Accuracy, F1, AUC | 불균형이면 Accuracy 함정 |
| 탐지 | mAP, Precision, Recall | IoU threshold |
| 생성 | FID, BLEU | 정량 + 정성 병행 |

안 되면?
- 과적합 (train↑ val↓) → Dropout, 증강, Early Stopping
- 과소적합 (train↓) → 모델 키우기, 학습 더, 피처 추가
- 성능 부족 → 데이터 더? 전처리 바꿔? 모델 바꿔?

### 실전 연습: "이 문제 어떻게 풀래?"

| # | 문제 | → 풀이 (문제→데이터→전처리→모델→평가) |
|---|------|-------------------------------------|
| 1 | 쇼핑몰 리뷰 긍부정 판별 | 이진분류 → 텍스트 → 토크나이징 → BERT → F1 |
| 2 | 공장 CCTV 안전모 미착용 실시간 감지 | 객체탐지 → 영상 → 프레임추출, 라벨링 → YOLO → mAP |
| 3 | 내일 주가 오를지 내릴지 | 분류/회귀 → 시계열 → 윈도우, 정규화 → LSTM → Acc/MSE |
| 4 | 운전자 표정으로 졸음/분노 탐지 | 다중분류 → 이미지+영상 → 얼굴crop, 정규화 → CNN+LSTM → F1 |
| 5 | 집값 예측 | 회귀 → 정형 → 스케일링, 결측치 → XGBoost → RMSE, R² |
| 6 | CT에서 종양 영역 픽셀 표시 | 세그멘테이션 → 의료이미지 → 정규화, resize → U-Net → Dice |
| 7 | 고객 문의 메일 5개 부서 분배 | 다중분류 → 텍스트 → 토크나이징 → BERT → F1 |

> **이 사고가 자동으로 되면 뭘 시켜도 할 수 있다.**
> 이 커리큘럼의 모든 단원은 이 흐름 위에서 진행된다.

## 1-3. 먼저 보자 — AI가 실제로 동작하는 모습

이론 들어가기 전에, **끝에 가면 이런 걸 만들 수 있다**는 걸 먼저 본다.

| 데모 | 뭘 보여주나 | 사용 기술 |
|------|-----------|----------|
| YOLO 웹캠 | 카메라 켜면 사람/물체를 실시간으로 잡음 | CNN + Detection |
| 챗봇 웹사이트 | 질문하면 AI가 대답하는 웹페이지 | Transformer + API |
| 감정 분류 | 문장 넣으면 긍정/부정 판별 | BERT Fine-tuning |
| 이미지 분류 | 사진 넣으면 고양이/개 구분 | CNN |
| 손글씨 인식 | 숫자 그리면 0~9 맞춤 | MLP/CNN |

→ "와 이런 게 되는구나" 느끼게 한 다음, **"근데 이게 어떻게 되는 건데?"** 로 들어간다.

---

# Part 2: 학습의 원리 — 오차를 줄이는 수학

Part 1에서 "AI = 오차를 줄이는 과정"이라고 했다.
그러면 오차를 **어떻게** 줄이는 건지, 가장 단순한 모델부터 손으로 직접 해본다.

## 2-1. 선형회귀 — AI의 가장 단순한 형태

**문제**: 공부시간(x)으로 점수(y) 예측
**모델**: y = wx + b (딱 이거)

여기서 배우는 것:
- Forward: 예측값 계산 (pred = wx + b)
- Loss: 오차 계산 (MSE = (pred - y)² / n)
- Backward: 오차를 줄이려면 w를 어디로 얼마나 움직여야 하나 (미분)
- Update: w = w - lr × gradient

**손계산**: w=0, b=0, lr=0.01에서 시작하여 **10 epoch** 종이에 계산
- 매 epoch마다: pred → loss → gradient(dw, db) → update
- Loss가 줄어드는 것을 직접 숫자로 확인

📝 **체크포인트**: 손계산 결과와 numpy 코드 결과가 소수점 4자리까지 일치하는가?

## 2-2. ANN (인공신경망) — 층을 쌓으면 더 복잡한 문제를 푼다

**이전 한계**: y = wx + b 는 직선만 그을 수 있음. XOR 같은 건 못 품.
**해결**: 뉴런을 여러 층으로 쌓자 → MLP (Multi-Layer Perceptron)

여기서 배우는 것:
- Perceptron 하나로 AND/OR 학습 → XOR은 실패 → 왜?
- MLP: Input → Hidden → Output (층을 쌓으면 비선형 문제 해결)
- **Chain Rule (역전파)**: 여러 층의 gradient를 어떻게 구하나
  - Loss → 출력층 gradient → 은닉층 gradient (체인처럼 연결)
- **손계산**: 2층 MLP의 Forward → Loss → Backward(Chain Rule) 전체 과정 3~10 epoch

실습 흐름:
1. numpy로 직접 구현 (손계산 검증)
2. 같은 것을 Keras 한 줄, PyTorch 한 줄로 → "아 이걸 프레임워크가 자동으로 해주는 거구나"
3. 다양한 데이터: Boston, Iris, Diabetes, DACON 따릉이 등

📝 **체크포인트**: 아무것도 안 보고 MLP 2층의 역전파를 Chain Rule로 종이에 전개할 수 있는가?

---

# Part 3: 데이터 유형별 모델 — 왜 모델이 여러 개 필요한가

ANN(MLP)은 만능이 아니다. **데이터 형태에 따라 더 잘하는 모델이 있다.**

```
MLP만으로는 부족한 이유:
- 이미지: 픽셀을 일렬로 펴면 공간 관계가 사라짐 → CNN
- 시계열/텍스트: 순서 정보를 모름 → RNN → LSTM
- 긴 시퀀스: RNN은 앞부분을 잊음 → LSTM → Attention → Transformer
```

이 Part에서는 "왜 이 모델이 필요한가?"부터 시작해서, 각 모델의 원리를 손으로 이해한다.

## 3-1. CNN — 이미지를 다루는 모델

**문제**: 이미지 분류 (손글씨, 얼굴, 물체 인식)
**MLP의 한계**: 28×28 이미지를 784개 숫자로 펴버리면 "어디에 뭐가 있는지" 정보가 날아감
**CNN의 해결**: 필터를 슬라이딩하면서 **공간적 특징**(엣지, 패턴)을 추출

여기서 배우는 것:
- Convolution 연산: 필터가 이미지 위를 슬라이딩하며 특징 추출
- **손계산**: 4×4 이미지에 3×3 필터 적용 → 출력의 각 값 직접 계산
- 출력 크기 공식: (Input - Kernel + 2×Padding) / Stride + 1
- Pooling: MaxPool로 크기 줄이면서 핵심 정보 유지
- CNN 파이프라인: Conv → ReLU → Pool → Conv → ReLU → Pool → Flatten → Dense
- 데이터: MNIST, Fashion-MNIST, CIFAR-10/100, Cat vs Dog

📝 **체크포인트**: 출력 크기 공식 5문제 연속 정답? Conv 연산을 손으로 할 수 있는가?

## 3-2. RNN — 순서가 있는 데이터를 다루는 모델

**문제**: 시계열 예측, 텍스트 처리 ("나는 오늘 ___")
**CNN/MLP의 한계**: 시간 순서를 전혀 모름. 어제와 오늘 데이터를 독립으로 봄.
**RNN의 해결**: **Hidden State**로 이전 정보를 다음 스텝에 전달

여기서 배우는 것:
- h_t = tanh(W_xh × x_t + W_hh × h_(t-1) + b) — 이전 기억 + 현재 입력
- **손계산**: h_0=[0,0]에서 시작하여 3 timestep의 hidden state 직접 계산
- **Vanishing Gradient 문제**: 긴 시퀀스에서 앞 정보를 잊는 이유
  - gradient = tanh' × W 가 계속 곱해짐 → 0으로 소멸
- 데이터: sin 함수 예측, Jena Climate, 주가

📝 **체크포인트**: "Vanishing Gradient가 왜 생기는지" 후배한테 수식 없이 설명할 수 있는가?

## 3-3. LSTM — RNN의 기억력 문제 해결

**이전 한계**: RNN은 긴 시퀀스에서 앞부분 정보를 잊어버림
**LSTM의 해결**: 3개의 **Gate**로 기억을 제어

| Gate | 역할 | 비유 |
|------|------|------|
| Forget Gate | 이전 기억 중 얼마나 버릴지 (0~1) | 필요 없는 기억 삭제 |
| Input Gate | 새 정보를 얼마나 받아들일지 (0~1) | 새 기억 저장 |
| Output Gate | cell state에서 얼마나 내보낼지 (0~1) | 필요한 것만 출력 |

- **Cell State**: gradient가 거의 그대로 흐르는 고속도로
- **손계산**: 각 gate 출력 → cell state 업데이트 → hidden state 과정
- RNN vs LSTM gradient 크기 비교 (50 timestep 후)
- 데이터: 시계열 예측, 텍스트 분류

📝 **체크포인트**: LSTM의 3개 Gate를 그림으로 그리고 각 역할을 설명할 수 있는가?

## ✏️ Part 2~3 종합 연습장

### 손코딩 시험 (아무것도 안 보고 종이에)
| # | 문제 | 목표 시간 |
|---|------|----------|
| 1 | Linear Regression forward + backward 1 epoch | 5분 |
| 2 | MLP 2층 Chain Rule 전개 | 10분 |
| 3 | CNN 출력 크기 계산 5문제 | 전부 정답 |
| 4 | RNN hidden state 3 timestep 계산 | 10분 |
| 5 | LSTM gate 1 timestep 계산 | 15분 |

### 목표 수치 (프레임워크 코드)
| 태스크 | 목표 |
|--------|------|
| MNIST MLP accuracy | ≥ 97% |
| CIFAR-10 CNN accuracy | ≥ 75% |
| sin 함수 LSTM 예측 MSE | ≤ 0.01 |

### "이 모델은 언제 쓰는 건데?" 정리 (반드시 자기 말로 작성)
| 모델 | 어떤 문제에 | 어떤 데이터에 | 왜 이 모델인지 |
|------|-----------|-------------|--------------|
| MLP | | | |
| CNN | | | |
| RNN | | | |
| LSTM | | | |

---

# Part 4: 현대 아키텍처 — Attention과 Transformer

Part 3에서 배운 모델들의 한계를 하나씩 해결해 나가는 과정.

```
LSTM만으로는 부족한 이유:
- 번역: "나는 학생이다" (3단어) → "I am a student" (4단어) → 입출력 길이 다름 → Seq2Seq
- Seq2Seq: 긴 문장을 벡터 하나에 우겨넣음 → 정보 손실 → Attention
- Attention: RNN에 붙여 썼음 → 순차처리라 느림 → Transformer (Attention만으로!)
```

## 4-1. Seq2Seq (Encoder-Decoder)

**문제**: 입출력 길이가 다른 문제 (번역, 요약, 챗봇)
**LSTM의 한계**: 입력 길이 = 출력 길이여야 함
**해결**: Encoder가 입력을 압축 → Decoder가 출력을 생성

- Encoder: 입력 시퀀스를 처리해서 **Context Vector** 하나로 압축
- Decoder: Context Vector에서 출력을 한 단어씩 생성
- Teacher Forcing: 학습 시 정답을 다음 입력으로 사용
- **한계**: 문장이 길면 Context Vector 하나에 정보가 다 안 들어감 (병목)

📝 **체크포인트**: Encoder-Decoder 구조를 그림으로 그리고 데이터 흐름을 설명할 수 있는가?

## 4-2. Attention

**이전 한계**: Seq2Seq의 Context Vector 병목
**해결**: Decoder가 매 스텝마다 Encoder의 **모든 hidden state**를 참조

- 핵심: "지금 이 단어를 생성할 때 원문의 어디를 봐야 하는지" 가중치를 계산
- Attention Score → softmax → Weight → Context Vector (가중 평균)
- **손계산**: 3개 토큰에 대한 Q·K 내적 → softmax → weighted sum
- Bahdanau (Additive) vs Luong (Dot-product)
- Attention Heatmap: 모델이 어디를 보고 있는지 시각화

📝 **체크포인트**: 3개 토큰 Attention Score → Weight → Context를 손으로 계산할 수 있는가?

## 4-3. Transformer

**이전 한계**: Attention을 RNN에 붙여 썼음 → 순차처리라 병렬화 불가, 느림
**해결**: "Attention is All You Need" — RNN 없이 Attention**만으로** 전부 처리

- Self-Attention: 같은 시퀀스 내에서 모든 단어가 서로를 참조 (병렬!)
- **손계산**: Q, K, V 행렬곱 → Scaled Dot-Product → Output
- Multi-Head Attention: 여러 관점(head)에서 동시에 관계를 봄
- Positional Encoding: 순서 정보가 없으니 sin/cos로 위치 주입
- Transformer Block: Self-Attention + Add&Norm + FFN + Add&Norm
- 이게 N번 쌓이면: BERT(12층), GPT-3(96층)

📝 **체크포인트**: Transformer Block 구조를 보지 않고 처음부터 끝까지 그릴 수 있는가?

## ✏️ Part 4 종합 연습장

### 손코딩 시험
| # | 문제 | 목표 시간 |
|---|------|----------|
| 1 | Seq2Seq Encoder 3 timestep → Context Vector | 10분 |
| 2 | Attention Score 3개 토큰 + softmax | 10분 |
| 3 | Self-Attention Q, K, V 전체 계산 | 15분 |
| 4 | Transformer Block 구조 그리기 | 보지 않고 완성 |

### 목표 수치
| 태스크 | 목표 |
|--------|------|
| Seq2Seq 숫자 뒤집기 accuracy | ≥ 90% |

### "왜 이 모델이 등장했는지" 정리 (반드시 자기 말로)
| 모델 | 이전 모델의 어떤 한계를 | 어떻게 해결했는지 |
|------|---------------------|-----------------|
| Seq2Seq | | |
| Attention | | |
| Transformer | | |

---

# Part 5: 실전 응용 — 과제에 바로 투입될 수 있는 스킬

Part 2~4에서 원리를 이해했으니, 이제 **실제 문제를 풀 수 있는 도구**를 익힌다.
여기서부터는 "이론을 알아" 수준이 아니라 **"혼자서 만들 수 있어"** 수준이 목표.

## 5-1. 텍스트 분류 & NLP 실전

**문제**: 리뷰 감정 분류, 뉴스 카테고리 분류
**사고 흐름**: 분류 → 텍스트 → 토크나이징 → BERT Fine-tuning → F1

- HuggingFace Pipeline: 한 줄로 감정분석 (먼저 결과를 보고 시작)
- BERT Fine-tuning: 커스텀 데이터로 분류기 학습하는 전체 과정
- 데이터: IMDB, 네이버 영화 리뷰, AG News, Reuters
- 평가: Accuracy, F1, Confusion Matrix
- 한국어: KoBERT, KcELECTRA

📝 **체크포인트**: BERT Fine-tuning 코드를 처음부터 혼자 작성할 수 있는가?

**목표 수치**:
| 태스크 | 목표 |
|--------|------|
| IMDB 감정분류 accuracy | ≥ 88% |
| 네이버 영화 리뷰 accuracy | ≥ 85% |

## 5-2. LLM API + 챗봇 웹사이트

**문제**: AI 챗봇 서비스를 만들고 싶다
**사고 흐름**: 생성 → 텍스트 → API 호출 → GPT/Claude → 사용자 피드백

- OpenAI API / Claude API 사용법
- System Prompt, Temperature, Token 개념
- Streamlit으로 챗봇 웹 UI 만들기
- Gradio로 ML 데모 페이지
- 스트리밍 응답 (한 글자씩 출력)
- 응용: RAG (문서 기반 질의응답), 코드 리뷰어 등

📝 **체크포인트**: API 연동 챗봇 웹사이트를 1시간 안에 혼자 만들 수 있는가?

## 5-3. YOLO 실시간 Object Detection

**문제**: 영상에서 특정 객체를 실시간으로 찾고 싶다
**사고 흐름**: 객체탐지 → 영상 → 라벨링 → YOLO → mAP

Detection 발전사 (각 모델이 이전의 어떤 한계를 해결했는지):
```
R-CNN (2014): Selective Search → CNN → SVM (느림, 47초/장)
    ↓ "전체 이미지를 한번에 보면 안 되나?"
Fast R-CNN (2015): 이미지 전체 CNN + RoI Pooling
    ↓ "Region Proposal도 학습시키면?"
Faster R-CNN (2015): RPN으로 End-to-end 학습
    ↓ "픽셀 단위로 잡으면?"
Mask R-CNN (2017): + Instance Segmentation
    ↓ "2단계 말고 한번에 하면 빠르지 않나?"
YOLO (2016~): 1단계 검출, 실시간 가능 (45+ FPS)
```

실습:
- YOLOv8 사전학습 모델로 즉시 사용
- 웹캠 실시간 탐지
- 커스텀 데이터셋: Roboflow 라벨링 → YOLO format → Fine-tuning
- 평가: mAP, Precision, Recall
- Export: ONNX, TensorRT (Jetson 배포용)

📝 **체크포인트**: 커스텀 데이터 100장 라벨링 → YOLO 학습 → 실시간 탐지까지 혼자 할 수 있는가?

**목표 수치**:
| 태스크 | 목표 |
|--------|------|
| 커스텀 데이터 mAP50 | ≥ 0.7 |

## ✏️ Part 5 종합 실전 테스트

이건 시험이 아니라 **"과제에 투입해도 되는지"** 확인:

| # | 테스트 | 제한 시간 | 합격 기준 |
|---|--------|----------|----------|
| 1 | 새 텍스트 데이터셋 → BERT 분류기 학습 + 평가 | 30분 | F1 ≥ 0.8 |
| 2 | API 챗봇 웹사이트 완성 + 동작 | 1시간 | 대화 가능 |
| 3 | 새 이미지 50장 라벨링 + YOLO 학습 + 웹캠 탐지 | 2시간 | 탐지 동작 |

---

# Part 6: 심화 — 과제 진행하면서 필요할 때

이건 순서대로 할 필요 없다. 과제나 논문에서 필요한 것부터 한다.

## 6-1. GAN (Generative Adversarial Network)
- Generator vs Discriminator 경쟁 학습
- 이미지 생성, Style Transfer
- 응용: 데이터 증강, 얼굴 생성

## 6-2. Detection 심화
- R-CNN 계열 직접 구현 (구조 이해용)
- Faster R-CNN with PyTorch (torchvision)
- Mask R-CNN: Instance Segmentation
- YOLO 커스텀 + Jetson AGX Thor 배포

## 6-3. 멀티모달 & 과제 연계
- Vision + Language (CLIP, BLIP)
- Audio + Vision (멀티모달 감정인식)
- 과제 연계: 카메라 + 음성 + 생체신호 → 운전자 감정 판단

---

# 부록

## 기존 레포 코드 활용

| 폴더 | 내용 | 커리큘럼 연계 |
|------|------|--------------|
| `keras/` (01~80) | TF2 기반 전체 파이프라인, 다양한 데이터셋 | Part 2~3 실습 |
| `keras2/` (68~80) | Optimizer, Transfer Learning, AutoKeras | Part 3~4 심화 |
| `ml/` (01~46) | scikit-learn (PCA, XGB, Ensemble 등) | ML 기초 보조 |
| `torch/` (01~22) | PyTorch MLP, CNN, RNN, LSTM | Part 2~3 프레임워크 비교 |

## 각 단원 진행 흐름

```
이 모델이 풀려는 문제가 뭔지? (Top-down: 큰 그림부터)
    ↓
이전 모델의 어떤 한계 때문에 나왔는지?
    ↓
데모: 실제로 동작하는 모습 먼저 보기
    ↓
핵심 구조와 수식 설명
    ↓
손계산 (종이 + 펜, 3~10 epoch)
    ↓
numpy 손코딩 (손계산 결과 검증)
    ↓
프레임워크 구현 (Keras / PyTorch)
    ↓
다양한 데이터셋으로 실험
    ↓
체크포인트 확인 (목표 수치 달성?)
    ↓
"이 모델은 언제 쓰는 건데?" 자기 말로 정리
```

## 데이터셋 목록

| 유형 | 데이터셋 |
|------|---------|
| 정형 | Boston, California Housing, Iris, Wine, Diabetes, DACON 따릉이 |
| 이미지 | MNIST, Fashion-MNIST, CIFAR-10/100, Cat vs Dog, 커스텀 |
| 텍스트 | IMDB, Reuters, 네이버 영화 리뷰, AG News |
| 시계열 | Jena Climate, 주가, sin 함수 |
| 객체탐지 | COCO, VOC, Roboflow 커스텀 |

## 필요 환경

```
numpy, matplotlib, pandas
tensorflow, keras
torch, torchvision
scikit-learn, xgboost
transformers, datasets (HuggingFace)
streamlit, gradio
openai, anthropic
ultralytics (YOLOv8), opencv-python
roboflow
```
