# 주차별 교육 계획 & 진도 관리

---

## 사용법

1. 매주 해당 주차의 **학습 → 실습 → 숙제**를 순서대로 진행
2. 숙제는 **다음 주 시작 전까지** 제출 (GitHub repo 또는 노션)
3. 매주 시작할 때 **지난주 체크리스트** 확인 → 통과 못한 항목은 보충 후 진행
4. 통과 기준을 못 넘기면 다음 주차로 넘어가지 않는다

---

# Phase 1: 기초 (Week 1~6)

---

## Week 1: 오리엔테이션 + AI란 무엇인가

### 학습 내용
- Part 0: 연구실 소개, 국책과제 구조, 진로, 공부법
- Part 1: AI 정의 — "오차를 줄이는 과정"
- Part 1: 문제 해결 사고 흐름 (문제→데이터→전처리→모델→평가)
- Part 2-1: 선형회귀 y=wx+b

### 실습
- `01_linear_regression_numpy.py` 실행 및 분석
- Loss가 줄어드는 과정을 matplotlib으로 시각화

### 숙제
| # | 과제 | 제출물 |
|---|------|--------|
| 1 | **손계산**: w=0, b=0, lr=0.01에서 y=2x+1 데이터로 **10 epoch** 종이에 계산 | 손계산 사진 (매 epoch의 pred, loss, dw, db, 새 w, b) |
| 2 | numpy 코드 실행 후 손계산과 결과 비교 스크린샷 | 소수점 4자리까지 일치 확인 |
| 3 | lr=0.001, 0.01, 0.1, 1.0 각각 돌려보고 loss 그래프 비교 | 그래프 4개 + 왜 그런지 설명 1줄씩 |
| 4 | `problem_solving_drill.md` 1~5번 풀기 | 문제→데이터→전처리→모델→평가 순서로 작성 |

### 통과 체크리스트
- [ ] "AI = 오차를 줄이는 과정"을 자기 말로 설명할 수 있다
- [ ] Forward → Loss → Gradient → Update 파이프라인을 그릴 수 있다
- [ ] 10 epoch 손계산 결과가 코드와 소수점 4자리까지 일치
- [ ] 문제 해결 사고 흐름 5문제 중 3개 이상 정답

---

## Week 2: ANN (Perceptron → MLP → 역전파)

### 학습 내용
- Part 2-2: Perceptron, AND/OR/XOR
- MLP: 층을 쌓으면 비선형 문제 해결
- Chain Rule을 이용한 역전파 전체 과정
- numpy vs Keras vs PyTorch 비교

### 실습
- `01_perceptron_numpy.py` — AND 학습, XOR 실패 확인
- `02_mlp_backprop_numpy.py` — XOR 해결, decision boundary 시각화
- `03_mlp_keras_pytorch.py` — 같은 것을 프레임워크로

### 숙제
| # | 과제 | 제출물 |
|---|------|--------|
| 1 | **손계산**: MLP 2층 (Input2→Hidden2→Output1), 입력 [1,0] 정답 1에 대해 Forward→Loss→Backward→Update **3 epoch** | 손계산 사진 (Chain Rule 전개 포함) |
| 2 | numpy 코드로 손계산 결과 검증 | 일치 확인 스크린샷 |
| 3 | Keras로 Boston 집값 예측 (회귀), Iris 꽃 분류 (다중분류) 각각 구현 | 코드 + 결과(loss, accuracy) |
| 4 | DACON 따릉이 데이터로 MLP 회귀 모델 만들기 | 코드 + RMSE 결과 |

### 통과 체크리스트
- [ ] XOR이 단일 퍼셉트론으로 안 되는 이유를 설명할 수 있다
- [ ] **안 보고** MLP 2층의 역전파를 Chain Rule로 종이에 전개할 수 있다
- [ ] "프레임워크 = 내가 손으로 한 수학을 자동화한 것"을 이해했다
- [ ] Keras/PyTorch 중 하나로 분류 모델을 혼자 짤 수 있다

---

## Week 3: CNN (이미지를 다루는 모델)

### 학습 내용
- Part 3-1: CNN
- 왜 MLP로 이미지가 안 되는지
- Convolution, Padding, Stride, Pooling
- 출력 크기 공식

### 실습
- `01_convolution_numpy.py` — 필터 연산 손코딩, 다양한 필터 적용
- Keras로 MNIST CNN 구현 (기존 코드 `keras/keras35~` 참고)
- CIFAR-10으로 확장

### 숙제
| # | 과제 | 제출물 |
|---|------|--------|
| 1 | **손계산**: 4×4 이미지에 3×3 필터, stride=1, padding=0 결과 계산 | 손계산 사진 |
| 2 | 출력 크기 공식 10문제 풀기 (practice_handcoding.md Level 3) | 풀이 사진 |
| 3 | MNIST CNN 구현 → accuracy **97% 이상** 달성 | 코드 + accuracy 스크린샷 |
| 4 | CIFAR-10 CNN 구현 → accuracy **75% 이상** 달성 | 코드 + accuracy 스크린샷 |
| 5 | Fashion-MNIST 또는 Cat vs Dog 중 하나 추가 실험 | 코드 + 결과 |

### 통과 체크리스트
- [ ] Conv 연산을 손으로 계산할 수 있다
- [ ] 출력 크기 공식 5문제 연속 정답
- [ ] MNIST accuracy ≥ 97%
- [ ] CIFAR-10 accuracy ≥ 75%
- [ ] CNN 파이프라인 (Conv→ReLU→Pool→...→Dense)을 그릴 수 있다

---

## Week 4: RNN + LSTM (순서가 있는 데이터)

### 학습 내용
- Part 3-2: RNN — Hidden State, Vanishing Gradient
- Part 3-3: LSTM — 3개 Gate, Cell State
- RNN vs LSTM 비교

### 실습
- `01_rnn_numpy.py` — hidden state 흐름 확인, vanishing gradient 시각화
- `01_lstm_gate_numpy.py` — gate 동작 확인
- Keras/PyTorch로 sin 함수 예측

### 숙제
| # | 과제 | 제출물 |
|---|------|--------|
| 1 | **손계산**: RNN h_t 3 timestep 계산 (practice_handcoding.md Level 4) | 손계산 사진 |
| 2 | **손계산**: LSTM 1 timestep gate 계산 (practice_handcoding.md Level 5) | 손계산 사진 |
| 3 | Vanishing Gradient: 0.65^50 계산하고 왜 문제인지 설명 | 계산 + 설명 |
| 4 | sin 함수 LSTM 예측 → MSE **0.01 이하** | 코드 + loss 그래프 + MSE |
| 5 | Jena Climate 또는 주가 데이터로 시계열 예측 | 코드 + 결과 |

### 통과 체크리스트
- [ ] RNN hidden state 3 timestep을 손으로 계산할 수 있다
- [ ] Vanishing Gradient를 후배한테 수식 없이 설명할 수 있다
- [ ] LSTM 3개 Gate 역할을 그림으로 그리고 설명할 수 있다
- [ ] sin LSTM MSE ≤ 0.01
- [ ] "이 모델은 언제 쓰는 건데?" 표를 MLP/CNN/RNN/LSTM 전부 작성

---

## Week 5: Phase 1 종합 점검

### 이번 주는 새로운 내용 없음. 복습 + 시험 주간.

### 손코딩 시험 (종이 + 펜, 제한 시간)
| # | 문제 | 제한 시간 | 통과 |
|---|------|----------|------|
| 1 | Linear Regression forward + backward 1 epoch | 5분 | [ ] |
| 2 | MLP 2층 Chain Rule 역전파 전개 | 10분 | [ ] |
| 3 | CNN 출력 크기 계산 5문제 | 5분 | [ ] |
| 4 | RNN hidden state 3 timestep | 10분 | [ ] |
| 5 | LSTM gate 1 timestep | 15분 | [ ] |

### 코딩 시험 (노트북, 제한 시간)
| # | 문제 | 제한 시간 | 합격 기준 |
|---|------|----------|----------|
| 1 | 새 정형 데이터셋 → MLP 회귀 모델 | 20분 | R² ≥ 0.7 |
| 2 | 새 이미지 데이터셋 → CNN 분류 모델 | 30분 | accuracy ≥ 80% |
| 3 | 새 시계열 데이터 → LSTM 예측 모델 | 30분 | MSE ≤ 0.05 |

### 숙제
| # | 과제 | 제출물 |
|---|------|--------|
| 1 | `checkpoint_part2_3.md` 모든 항목 체크하고 제출 | 체크리스트 |
| 2 | "이 모델은 언제 쓰는 건데?" 표 완성 (MLP, CNN, RNN, LSTM) | 자기 말로 작성 |
| 3 | `problem_solving_drill.md` 6~10번 풀기 | 풀이 |

### 통과 기준
- 손코딩 시험 5개 중 **4개 이상** 통과
- 코딩 시험 3개 중 **2개 이상** 합격
- 못 넘기면 Week 5 반복

---

## Week 6: Seq2Seq + Attention

### 학습 내용
- Part 4-1: Seq2Seq — Encoder-Decoder, Context Vector 병목
- Part 4-2: Attention — Score, Weight, Context

### 실습
- `01_seq2seq_numpy.py` — Encoder → Context → Decoder 흐름 확인
- `01_attention_numpy.py` — Attention heatmap 시각화
- Keras로 간단한 Seq2Seq 번역 구현

### 숙제
| # | 과제 | 제출물 |
|---|------|--------|
| 1 | **손계산**: Encoder 3 timestep → Context Vector (practice_handcoding.md Level 6-1) | 손계산 사진 |
| 2 | **손계산**: Attention Score 3개 토큰 + softmax + Context Vector | 손계산 사진 |
| 3 | Seq2Seq 숫자 뒤집기 구현 → accuracy **90% 이상** | 코드 + 결과 |
| 4 | "Context Vector 병목이 뭔지, Attention이 어떻게 해결하는지" 설명 글 작성 | A4 반 페이지 |

### 통과 체크리스트
- [ ] Encoder-Decoder 구조를 그림으로 그릴 수 있다
- [ ] Context Vector 병목을 설명할 수 있다
- [ ] Attention Score → Weight → Context를 손으로 계산할 수 있다
- [ ] Seq2Seq 숫자 뒤집기 accuracy ≥ 90%

---

# Phase 2: 현대 아키텍처 + 실전 (Week 7~12)

---

## Week 7: Transformer

### 학습 내용
- Part 4-3: Self-Attention, Multi-Head, Positional Encoding
- Transformer Block 전체 구조
- "Attention is All You Need" 핵심 정리

### 실습
- `01_self_attention_numpy.py` — Q, K, V 계산, Multi-Head, PE
- PyTorch nn.TransformerEncoder 사용해보기

### 숙제
| # | 과제 | 제출물 |
|---|------|--------|
| 1 | **손계산**: Self-Attention Q, K, V 전체 (practice_handcoding.md Level 6-2) | 손계산 사진 |
| 2 | Transformer Block 구조를 **안 보고** 종이에 그리기 | 그림 사진 |
| 3 | `checkpoint_part4.md` 모든 항목 체크 | 체크리스트 |
| 4 | "왜 이 모델이 등장했는지" 표 완성 (Seq2Seq, Attention, Transformer) | 자기 말로 작성 |
| 5 | `problem_solving_drill.md` 11~15번 풀기 | 풀이 |

### 통과 체크리스트
- [ ] Self-Attention Q, K, V 계산을 손으로 할 수 있다
- [ ] Scaling (√d_k) 하는 이유를 설명할 수 있다
- [ ] Multi-Head Attention의 필요성을 설명할 수 있다
- [ ] Transformer Block 전체 구조를 **보지 않고** 그릴 수 있다

---

## Week 8: 텍스트 분류 (BERT Fine-tuning)

### 학습 내용
- Part 5-1: HuggingFace, BERT Fine-tuning
- 텍스트 데이터 전처리: 토크나이저, 패딩
- 평가: Accuracy, F1, Confusion Matrix

### 실습
- `01_text_classification_huggingface.py` 실행
- IMDB 감정분류 Fine-tuning
- 네이버 영화 리뷰 한국어 분류

### 숙제
| # | 과제 | 제출물 |
|---|------|--------|
| 1 | IMDB 감정분류 → accuracy **88% 이상** | 코드 + 결과 + Confusion Matrix |
| 2 | 네이버 영화 리뷰 감정분류 → accuracy **85% 이상** | 코드 + 결과 |
| 3 | AG News (4 카테고리) 분류 구현 | 코드 + F1 score |
| 4 | 학습 데이터 수 (100, 500, 1000, 5000)별 성능 변화 그래프 | 그래프 + 분석 |

### 통과 체크리스트
- [ ] IMDB accuracy ≥ 88%
- [ ] 네이버 영화 리뷰 accuracy ≥ 85%
- [ ] BERT Fine-tuning 코드를 처음부터 혼자 작성할 수 있다
- [ ] F1 score와 Accuracy의 차이를 설명할 수 있다

---

## Week 9: LLM API + 챗봇 웹사이트

### 학습 내용
- Part 5-2: OpenAI/Claude API, System Prompt, Temperature
- Streamlit, Gradio

### 실습
- `01_chatbot_streamlit.py` 실행
- `02_chatbot_gradio.py` 실행
- 직접 System Prompt 바꿔가며 실험

### 숙제
| # | 과제 | 제출물 |
|---|------|--------|
| 1 | Streamlit 챗봇을 직접 만들기 (코드 안 보고) | 코드 + 동작 스크린샷 |
| 2 | System Prompt 3가지 변형 (코딩 도우미, 영어 선생님, 감정 분석기) | 각 대화 스크린샷 |
| 3 | 스트리밍 응답 (한 글자씩 출력) 구현 | 코드 + 동작 영상/gif |
| 4 | (도전) RAG: 텍스트 파일 업로드 후 질문에 답하는 챗봇 | 코드 + 데모 |

### 통과 체크리스트
- [ ] API 키 설정 + API 호출 코드를 혼자 작성할 수 있다
- [ ] System Prompt, Temperature, Token의 역할을 설명할 수 있다
- [ ] **1시간 안에** 챗봇 웹사이트를 처음부터 혼자 완성할 수 있다

---

## Week 10: YOLO 실시간 Detection

### 학습 내용
- Part 5-3: Detection 발전사, YOLOv8
- 사전학습 모델 사용, 웹캠 실시간, 커스텀 학습

### 실습
- `01_yolo_quickstart.py` — 사전학습 모델로 탐지
- `02_yolo_custom_training.py` — 커스텀 학습 구조 이해
- 웹캠 실시간 탐지 실행

### 숙제
| # | 과제 | 제출물 |
|---|------|--------|
| 1 | Detection 발전사 정리 (R-CNN→Fast→Faster→Mask→YOLO, 각 모델의 한계와 해결) | 정리 문서 |
| 2 | YOLOv8n/s/m 모델 비교 (속도 vs 정확도) | 비교 표 + 분석 |
| 3 | Roboflow에서 데이터 라벨링 **100장** + YOLO 커스텀 학습 | 라벨링 스크린샷 + 학습 결과 |
| 4 | 커스텀 모델로 웹캠 실시간 탐지 | 동작 영상/gif |

### 통과 체크리스트
- [ ] R-CNN → YOLO 발전 과정에서 각 모델의 핵심 개선점을 설명할 수 있다
- [ ] 커스텀 데이터 mAP50 ≥ 0.7
- [ ] **혼자서** 라벨링 → 학습 → 실시간 탐지 파이프라인을 완성할 수 있다

---

## Week 11: Phase 2 종합 점검

### 실전 테스트 (제한 시간)

| # | 테스트 | 제한 시간 | 합격 기준 |
|---|--------|----------|----------|
| 1 | 새 텍스트 데이터셋 → BERT 분류기 학습 + 평가 | 30분 | F1 ≥ 0.8 |
| 2 | API 챗봇 웹사이트 처음부터 완성 | 1시간 | 대화 가능 + 스트리밍 |
| 3 | 새 이미지 50장 라벨링 + YOLO 학습 + 웹캠 탐지 | 2시간 | 탐지 동작 |

### 숙제
| # | 과제 | 제출물 |
|---|------|--------|
| 1 | `checkpoint_part5.md` 모든 항목 체크 | 체크리스트 |
| 2 | `problem_solving_drill.md` 전체 15문제 다시 풀기 (정답 안 보고) | 풀이 |
| 3 | 자기가 만든 것 중 하나를 골라 5분 발표 준비 | 발표 자료 |

### 통과 기준
- 실전 테스트 3개 중 **2개 이상** 합격
- 문제풀이 15문제 중 **12개 이상** 정답
- 못 넘기면 Week 11 반복

---

## Week 12: 최종 발표 + 과제 투입 준비

### 최종 발표 (1인당 10분)

각자 아래 중 하나를 골라 **처음부터 끝까지 혼자 만들고 발표**:

| 프로젝트 | 난이도 | 포함해야 할 것 |
|----------|--------|---------------|
| A. 감정 분석 서비스 | 중 | BERT 분류 + Streamlit 웹 UI |
| B. 실시간 객체 탐지 | 중 | YOLO 커스텀 학습 + 웹캠 데모 |
| C. AI 챗봇 서비스 | 중 | API + RAG(문서 검색) + 웹 UI |
| D. 자유 주제 | 상 | 과제 관련 또는 본인 관심 분야 |

### 발표 평가 기준
| 항목 | 배점 | 기준 |
|------|------|------|
| 문제 정의 | 20% | 문제→데이터→전처리→모델→평가 사고 흐름이 명확한가 |
| 구현 완성도 | 30% | 실제로 동작하는가, 코드가 깔끔한가 |
| 결과 분석 | 20% | 성능 수치를 분석하고 개선점을 아는가 |
| 원리 이해 | 20% | "이 모델이 왜 이렇게 동작하는지" 질문에 답할 수 있는가 |
| 발표력 | 10% | 명확하게 설명하는가 |

### 과제 투입 판단
| 등급 | 기준 | 판단 |
|------|------|------|
| A | 모든 체크포인트 통과 + 발표 우수 | 즉시 투입 가능 |
| B | 대부분 통과 + 발표 보통 | 선배와 페어로 투입 |
| C | 일부 미통과 | 부족한 Part 보충 후 재평가 |

---

# 진도 추적표

학생 이름: ________________

## 주차별 진도

| 주차 | 주제 | 숙제 제출 | 체크리스트 통과 | 비고 |
|------|------|----------|---------------|------|
| W1 | AI 정의 + 선형회귀 | [ ] | [ ] | |
| W2 | ANN (MLP + 역전파) | [ ] | [ ] | |
| W3 | CNN | [ ] | [ ] | |
| W4 | RNN + LSTM | [ ] | [ ] | |
| W5 | **Phase 1 시험** | [ ] 손코딩 /5 | [ ] 코딩 /3 | |
| W6 | Seq2Seq + Attention | [ ] | [ ] | |
| W7 | Transformer | [ ] | [ ] | |
| W8 | BERT 텍스트 분류 | [ ] | [ ] | |
| W9 | API 챗봇 웹 | [ ] | [ ] | |
| W10 | YOLO Detection | [ ] | [ ] | |
| W11 | **Phase 2 시험** | [ ] 실전 /3 | [ ] 문제풀이 /15 | |
| W12 | **최종 발표** | [ ] | 등급: ___ | |

## 목표 수치 달성

| 태스크 | 목표 | 달성 | 날짜 |
|--------|------|------|------|
| 10 epoch 손계산 = numpy | 소수점 4자리 일치 | [ ] | |
| MLP Chain Rule 안 보고 전개 | 10분 이내 | [ ] | |
| CNN 출력 크기 5문제 | 전부 정답 | [ ] | |
| MNIST MLP accuracy | ≥ 97% | ___% | |
| CIFAR-10 CNN accuracy | ≥ 75% | ___% | |
| sin LSTM MSE | ≤ 0.01 | ___ | |
| Transformer Block 그리기 | 안 보고 완성 | [ ] | |
| IMDB BERT accuracy | ≥ 88% | ___% | |
| 네이버 리뷰 accuracy | ≥ 85% | ___% | |
| 챗봇 웹 1시간 완성 | 동작 확인 | [ ] | |
| 커스텀 YOLO mAP50 | ≥ 0.7 | ___ | |
| 문제풀이 15문제 | ≥ 12개 정답 | ___/15 | |

## 메모

| 날짜 | 내용 |
|------|------|
| | |
| | |
| | |
