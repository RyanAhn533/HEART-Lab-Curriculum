# HEART Lab - AI Education Curriculum

**세종대학교 HEART Lab 학석사 신입 연구원을 위한 AI 교육 커리큘럼**

> **인공지능 = Prediction(예측)과 Ground Truth(실제값)의 오차를 줄여나가는 과정**

---

## 교육 과정 구조

```
[입학/합류]
    ↓
[진단 테스트] ← 현재 수준 파악, 부족 영역 확인
    ↓
┌─────────────────────┐     ┌──────────────────────────┐
│  undergraduate/      │     │  masters/                 │
│  학석사 신입생 (12주)  │ ──→ │  석사 Condition (3개월)    │
│                     │     │                          │
│  모델 구현 능력      │     │  논문 구현+커스터마이징 능력 │
│  매월 월간 발표      │     │  매월 월간 발표 (평가)     │
└─────────────────────┘     └──────────────────────────┘
    ↓                           ↓
[과제 투입] ←───────────────────┘
```

---

## 폴더 구조

```
HEART-Lab-Curriculum/
│
├── README.md                ← 이 파일
├── DIAGNOSTIC_TEST.md       ← 진단 테스트 (40문제, 부족 영역 → 보충 매핑)
├── MANUAL.md                ← 교육 운영 매뉴얼 (교육자용)
│
├── undergraduate/           ← 학석사 신입생 교육 (12주)
│   ├── README.md
│   ├── CURRICULUM.md        ← 전체 커리큘럼 (Part 0~6, Top-down)
│   ├── WEEKLY_PLAN.md       ← 12주 주차별 계획 + 숙제 + 시험 + 진도표
│   ├── Part0_orientation/   ← 오리엔테이션
│   ├── Part2_learning_principle/  ← 선형회귀, ANN (손계산)
│   ├── Part3_data_models/   ← CNN, RNN, LSTM (손계산)
│   ├── Part4_modern/        ← Seq2Seq, Attention, Transformer
│   ├── Part5_practical/     ← BERT, 챗봇 웹, YOLO
│   ├── Part6_advanced/      ← GAN, Detection 심화, 멀티모달
│   └── worksheets/          ← 체크포인트, 손코딩 연습, 문제풀이 15제
│
└── masters/                 ← 석사 과정생 Condition (3개월)
    ├── README.md
    └── MASTERS_TRACK.md     ← 3개월 Condition + 월간 발표 + 평가
```

---

## 두 과정 비교

| | 학석사 신입생 (undergraduate) | 석사 Condition (masters) |
|---|---|---|
| **기간** | 12주 | 3개월 (12주) |
| **목표** | 모델을 구현할 수 있다 | 논문을 구현하고 커스터마이징할 수 있다 |
| **핵심 활동** | 손계산 → numpy → 프레임워크 | 논문 읽기 → 재현 → 수정 → 독립 연구 |
| **평가** | W5 시험, W11 시험, W12 최종 발표 | 매월 월간 발표 (1차→2차→3차) |
| **결과물** | 코드, 데모 | 실험 보고서, 논문 초고, 과제 기여 |
| **통과 후** | 석사 트랙 or 과제 투입 | 정식 연구원으로 과제 투입 |

---

## 월간 발표 제도

학부/석사 모두 **매달 랩미팅에서 발표**.

| 대상 | 발표 시간 | 내용 |
|------|----------|------|
| 신입생 | 10분 | 이번 달 학습 현황 + 결과물 |
| 석사 Condition | 15~25분 | 논문 재현/실험 진행 + 평가 |
| 정식 연구원 | 15분 | 연구 진행 + 실험 결과 + 다음 달 계획 |

---

## Quick Start

### 신입생이라면
1. `DIAGNOSTIC_TEST.md`로 현재 수준 진단
2. `undergraduate/CURRICULUM.md` 읽기
3. `undergraduate/WEEKLY_PLAN.md` Week 1부터 시작

### 석사 과정생이라면
1. `DIAGNOSTIC_TEST.md`로 현재 수준 진단
2. 부족한 부분이 있으면 `undergraduate/` 해당 Part 보충
3. `masters/MASTERS_TRACK.md` Month 1부터 시작

### 교육 담당자라면
1. `MANUAL.md` 읽기
2. `DIAGNOSTIC_TEST.md`로 학생 수준 파악
3. 해당 과정으로 안내

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
