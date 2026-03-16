# HEART Lab - AI Education Curriculum

**세종대학교 HEART Lab 학석사 신입 연구원을 위한 AI 교육 커리큘럼**

> **인공지능 = Prediction(예측)과 Ground Truth(실제값)의 오차를 줄여나가는 과정**

---

## Overview

| Part | 주제 | 핵심 |
|------|------|------|
| **Part 0** | 오리엔테이션 | 연구실 소개, 국책과제 구조, 진로, 공부법 |
| **Part 1** | AI의 전체 그림 | AI 정의, 문제 해결 사고 흐름, 데모 |
| **Part 2** | 학습의 원리 | 선형회귀, ANN, Chain Rule 역전파 (손계산) |
| **Part 3** | 데이터 유형별 모델 | CNN (이미지), RNN (시계열), LSTM (장기 기억) |
| **Part 4** | 현대 아키텍처 | Seq2Seq, Attention, Transformer (손계산) |
| **Part 5** | 실전 응용 | BERT 분류, API 챗봇 웹, YOLO 실시간 탐지 |
| **Part 6** | 심화 | GAN, Detection 심화, 멀티모달 |

---

## 핵심 사고 프레임워크

모든 AI 문제는 이 순서로 접근한다:

```
문제가 뭐지? → 데이터를 어떻게 쓰지? → 전처리는? → 어떤 모델? → 평가는?
```

모델 이름을 많이 아는 게 실력이 아니다. **문제를 보고 풀이 방법이 바로 떠오르는 것**이 실력이다.

---

## 폴더 구조

```
├── CURRICULUM.md                        # 전체 커리큘럼 문서 (필독)
│
├── Part0_orientation/                   # 연구실, 국책과제, 진로, 공부법
│
├── Part2_learning_principle/            # 학습의 원리 (손계산 중심)
│   ├── 2-1_linear_regression/           #   y=wx+b, Loss, Gradient Descent
│   │   └── 01_linear_regression_numpy.py
│   └── 2-2_ann/                         #   Perceptron, MLP, Chain Rule
│       ├── 01_perceptron_numpy.py
│       ├── 02_mlp_backprop_numpy.py
│       └── 03_mlp_keras_pytorch.py
│
├── Part3_data_models/                   # 데이터 유형별 모델
│   ├── 3-1_cnn/                         #   이미지 → Convolution
│   │   └── 01_convolution_numpy.py
│   ├── 3-2_rnn/                         #   시계열 → Hidden State
│   │   └── 01_rnn_numpy.py
│   └── 3-3_lstm/                        #   기억력 문제 → Gate
│       └── 01_lstm_gate_numpy.py
│
├── Part4_modern/                        # 현대 아키텍처
│   ├── 4-1_seq2seq/                     #   Encoder-Decoder
│   │   └── 01_seq2seq_numpy.py
│   ├── 4-2_attention/                   #   QKV, Attention Heatmap
│   │   └── 01_attention_numpy.py
│   └── 4-3_transformer/                #   Self-Attention, Multi-Head
│       └── 01_self_attention_numpy.py
│
├── Part5_practical/                     # 실전 응용 (과제 투입 스킬)
│   ├── 5-1_text_classification/         #   BERT Fine-tuning
│   │   └── 01_text_classification_huggingface.py
│   ├── 5-2_chatbot_web/                #   API 챗봇 웹사이트
│   │   ├── 01_chatbot_streamlit.py
│   │   └── 02_chatbot_gradio.py
│   └── 5-3_yolo/                       #   YOLOv8 실시간 탐지
│       ├── 01_yolo_quickstart.py
│       └── 02_yolo_custom_training.py
│
├── Part6_advanced/                      # 심화 (GAN, Detection, 멀티모달)
│
└── worksheets/                          # 연습장 & 체크리스트
    ├── checkpoint_part2_3.md            #   Part 2~3 자기 점검표
    ├── checkpoint_part4.md              #   Part 4 자기 점검표
    ├── checkpoint_part5.md              #   Part 5 실전 테스트
    ├── practice_handcoding.md           #   손코딩 연습 문제 (Level 1~6)
    └── problem_solving_drill.md         #   "이 문제 어떻게 풀래?" 15문제
```

---

## 교육 방법

### Top-down 학습

매 단원은 이 흐름으로 진행:

```
이 모델이 풀려는 문제가 뭔지?
    ↓
이전 모델의 한계 → 이 모델의 해결
    ↓
데모로 동작 먼저 보기
    ↓
핵심 수식 이해
    ↓
손계산 (종이, 3~10 epoch)
    ↓
numpy 손코딩 (검증)
    ↓
Keras / PyTorch 구현
    ↓
다양한 데이터셋 실험
    ↓
체크포인트 확인
```

### 손계산 규칙

1. 새 모델마다 **종이에 forward → loss → backward 최소 3 epoch**
2. Chain Rule을 **수식으로** 완전히 전개
3. 10 epoch까지 w, b 값 추적 → "학습이 되는 과정" 체감
4. numpy로 검증 → 그 후에야 프레임워크 사용

### 목표 수치

| 태스크 | 목표 |
|--------|------|
| MNIST MLP accuracy | >= 97% |
| CIFAR-10 CNN accuracy | >= 75% |
| sin LSTM MSE | <= 0.01 |
| IMDB BERT accuracy | >= 88% |
| 커스텀 YOLO mAP50 | >= 0.7 |

---

## 사용 데이터셋

| 유형 | 데이터셋 |
|------|---------|
| 정형 | Boston, California Housing, Iris, Wine, Diabetes, DACON |
| 이미지 | MNIST, Fashion-MNIST, CIFAR-10/100, Cat vs Dog |
| 텍스트 | IMDB, Reuters, 네이버 영화 리뷰, AG News |
| 시계열 | Jena Climate, 주가, sin 함수 |
| 객체탐지 | COCO, VOC, Roboflow 커스텀 |

---

## 환경 설정

```bash
pip install numpy matplotlib pandas
pip install tensorflow torch torchvision
pip install scikit-learn xgboost
pip install transformers datasets
pip install streamlit gradio
pip install openai anthropic
pip install ultralytics opencv-python
pip install roboflow
```

---

## License

이 커리큘럼은 HEART Lab 내부 교육용으로 제작되었습니다.
