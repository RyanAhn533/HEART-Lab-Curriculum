# 학석사 신입생 교육 과정 (12주)

> **목표: 과제에 투입되었을 때 바로 구현할 수 있는 수준**

## 교육 기간
12주 (약 3개월)

## 전체 구조

| Part | 주제 | 주차 |
|------|------|------|
| Part 0 | 오리엔테이션 (연구실, 국책과제, 진로, 공부법) | W1 |
| Part 1 | AI 전체 그림 (정의, 문제 해결 사고 흐름) | W1 |
| Part 2 | 학습의 원리 — 선형회귀, ANN, Chain Rule (손계산) | W1~W2 |
| Part 3 | 데이터 유형별 모델 — CNN, RNN, LSTM (손계산) | W3~W4 |
| Part 4 | 현대 아키텍처 — Seq2Seq, Attention, Transformer | W6~W7 |
| Part 5 | 실전 응용 — BERT, 챗봇 웹, YOLO | W8~W10 |
| Part 6 | 심화 — GAN, Detection 심화, 멀티모달 | 필요시 |

## 평가
- W5: Phase 1 종합 시험 (손코딩 + 코딩)
- W11: Phase 2 실전 테스트
- W12: 최종 발표 + 과제 투입 판단 (A/B/C 등급)
- 매월 월간 발표 (10분)

## 문서

| 파일 | 내용 |
|------|------|
| [CURRICULUM.md](CURRICULUM.md) | 전체 커리큘럼 (Part 0~6) |
| [WEEKLY_PLAN.md](WEEKLY_PLAN.md) | 12주 주차별 계획, 숙제, 시험, 진도표 |
| [worksheets/](worksheets/) | 체크포인트, 손코딩 연습, 문제풀이 15제 |

## 폴더 구조

```
undergraduate/
├── CURRICULUM.md
├── WEEKLY_PLAN.md
├── Part0_orientation/
├── Part2_learning_principle/
│   ├── 2-1_linear_regression/
│   └── 2-2_ann/
├── Part3_data_models/
│   ├── 3-1_cnn/
│   ├── 3-2_rnn/
│   └── 3-3_lstm/
├── Part4_modern/
│   ├── 4-1_seq2seq/
│   ├── 4-2_attention/
│   └── 4-3_transformer/
├── Part5_practical/
│   ├── 5-1_text_classification/
│   ├── 5-2_chatbot_web/
│   └── 5-3_yolo/
├── Part6_advanced/
└── worksheets/
    ├── checkpoint_part2_3.md
    ├── checkpoint_part4.md
    ├── checkpoint_part5.md
    ├── practice_handcoding.md
    └── problem_solving_drill.md
```
