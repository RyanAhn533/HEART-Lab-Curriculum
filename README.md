# HEART Lab AI Curriculum

**세종대학교 HEART Lab 신입 연구원을 위한 AI 교육 커리큘럼**

> **"AI로 문제를 풀 수 있는 사람을 만든다"**

---

## 교육 철학

```
통계 (1800s) → 머신러닝 (1950s~) → 딥러닝 (2012~)
```

실제 발전 순서대로, **데이터를 이해하고 → 예측하고 → 분류하고 → 모델을 확장**하는 흐름.
모든 코드는 동일한 **#0→#1→#2→#3→#4** 구조로 작성되어, 새 개념이 추가될 때
**"이번엔 뭐가 달라졌는지"** 에만 집중할 수 있다.

```python
#0. 라이브러리
#1. 데이터
#2. 데이터 전처리 및 분할
#3. 모델              ← 여기만 바뀐다
#4. 평가
```

---

## 커리큘럼 구조

```
Chapter 1:   통계와 데이터           ─── "데이터가 어떻게 생겼는지 본다"
Chapter 2:   예측과 분류의 기초       ─── "직선으로 예측하고, 분류한다"
Chapter 3:   머신러닝 모델들          ─── "다양한 모델을 배우고 비교한다"
Chapter 3.5: 수학 근육 부트캠프 ★     ─── "역전파를 손으로 계산해 본다" (Ch.4의 벽 넘기)
Chapter 4:   딥러닝 입문 (TF2/Keras) ─── "신경망으로 확장한다"
                                     (Chapter 5~10 개발 중 — 손계산 워크시트는 worksheets/에 먼저 준비됨)
```

> 전체 지도는 **[ROADMAP.md](ROADMAP.md)**, 논문 읽는 법은 **[PAPER_GUIDE.md](PAPER_GUIDE.md)** 참고.

| Chapter | 주제 | 항목 수 | 프레임워크 |
|---------|------|--------|-----------|
| **1. 통계와 데이터** | 기술통계, 확률분포, EDA, 전처리 | 5개 | numpy, pandas |
| **2. 예측과 분류** | 선형회귀, 경사하강법, 로지스틱, 평가지표 | 8개 | sklearn |
| **3. 머신러닝** | 결정트리, RF, SVM, KNN, K-Means, PCA | 8개 | sklearn |
| **3.5 수학 부트캠프** | 손계산: 미분, 연쇄법칙+계산그래프, 선형대수, XOR | 5개 | 종이 + numpy |
| **4. 딥러닝** | 뉴런, 활성화, TF2 회귀/분류, EarlyStopping | 8개 | TF2/Keras |

---

## 8주 진행 계획 — 풀타임(9-to-6) 기준, 5주 커리큘럼 + 3주 프로젝트

| 주차 | 내용 | 관문 |
|------|------|------|
| **W1** | 오리엔테이션 + 진단 → Ch.1 + B-4 → Ch.2 전반 (2-1~2-4) | GATE 1·2 |
| **W2** | Ch.2 후반 (2-5~2-8) → Ch.3 전체 | ★ 시험 1 |
| **W3** | 수학 부트캠프 (Ch.3.5) → Ch.4 + 4-7 혼자 짜보기 | GATE 3.5·4 |
| **W4** | CNN 손계산+실습 → RNN/LSTM 손계산+실습 (AI5 레포) | — |
| **W5** | 논문 4편 → ★ 시험 2 → 보충 버퍼 | ★ 시험 2 |
| **W6~7** | 미니 연구 프로젝트 (처음 보는 데이터 end-to-end) | 중간 점검 |
| **W8** | 프로젝트 마무리 + ★ 최종 발표 → 과제 투입 판정 | ★ 발표 |

> 일정 상세는 [ROADMAP.md](ROADMAP.md), 관문·시험 스펙은 [CHECKPOINTS.md](CHECKPOINTS.md).
> 파트타임(수업 병행)으로 돌리면 같은 내용이 12~16주.

---

## 대상별 시작점

| 대상 | 시작 | 예상 기간 (풀타임) |
|------|------|----------|
| 파이썬 안 됨 (진단 Section C 불통과) | **Week 0: 파이썬 부트캠프 1주** → Chapter 1 | 9주 |
| 학부 3~4학년 / 석사 신입 (표준) | Chapter 1부터 | 8주 |
| ML 경험자 (진단 수학·코딩 80%↑) | Chapter 3 + 부트캠프 확인 후 Ch.4 | 4~5주 |

---

## 각 항목 구성

모든 항목은 **이론 + 실습** 세트:

```
[번호]_[주제].md     ← 이론 (개념, 직관, 시각 예시, 레퍼런스)
[번호]_[주제].py     ← 실습 (#0→#1→#2→#3→#4 구조, 60~120줄)
```

코드 헤더에 **이전과 차이**, **왜 배우는가**, **보고 오기** 명시:

```python
# ============================================
# 3-1. 결정트리 (Decision Tree) — Iris
# 이전(2-8)과 차이: #3에 sklearn DecisionTreeClassifier
#
# 왜 배우는가:
#   질문을 반복해서 분류. 해석 가능한 모델.
#
# ▶ 보고 오기: StatQuest "Decision Trees"
# ============================================
```

---

## 이론 학습 방법

```
영상 (직관) → 블로그 리뷰 (정리) → 논문 (원본)
```

| 순위 | 리소스 | 용도 |
|------|--------|------|
| 1 | **3Blue1Brown** (한글자막) | 수학/직관 최강 |
| 2 | **구글 "[주제] 논문 리뷰"** | 한국어 블로그로 핵심 파악 |
| 3 | **StatQuest** | 통계/ML 기초 |
| 4 | **codingopera.tistory.com** | Transformer 계열 |

### 위키독스 (WikiDocs) — 한국어 무료 교재

| 교재 | 링크 | 커리큘럼 연결 |
|------|------|-------------|
| **점프 투 파이썬** | [wikidocs.net/book/1](https://wikidocs.net/book/1) | 파이썬 기초 (사전 준비) |
| **numpy/pandas/matplotlib 기초** | [wikidocs.net/32829](https://wikidocs.net/32829) | Chapter 1 데이터 도구 |
| **데이터 분석 3종 패키지** | [wikidocs.net/21047](https://wikidocs.net/21047) | Chapter 1 데이터 도구 |
| **파이썬으로 데이터 다루기 기초** | [wikidocs.net/book/9306](https://wikidocs.net/book/9306) | Chapter 1~2 numpy/pandas/sklearn |
| **Python 강좌와 통계** | [wikidocs.net/book/15702](https://wikidocs.net/book/15702) | Chapter 1 통계 보충 |
| **토닥토닥 sklearn 머신러닝** | [wikidocs.net/book/2383](https://wikidocs.net/book/2383) | Chapter 2~3 sklearn ML |
| **인공지능(AI) & 머신러닝(ML) 사전** | [wikidocs.net/book/5942](https://wikidocs.net/book/5942) | 전체 용어/개념 참고 |
| **토닥토닥 딥러닝 (텐서플로 v2)** | [wikidocs.net/book/4172](https://wikidocs.net/book/4172) | Chapter 4 딥러닝 |
| **딥러닝 파이토치 교과서** | [wikidocs.net/book/2788](https://wikidocs.net/book/2788) | Chapter 10 PyTorch |
| **딥러닝을 이용한 자연어 처리 입문** | [wikidocs.net/book/2155](https://wikidocs.net/book/2155) | Chapter 7 NLP |

> 위키독스는 **무료**이고 **한국어**라 비전공자가 접근하기 좋다. 영상(3B1B)으로 직관 잡고, 위키독스로 코드 실습하는 조합 추천.

---

## 레퍼런스

이 커리큘럼은 아래 교육과정을 참고하여 설계:

- [Stanford CS229: Machine Learning](https://cs229.stanford.edu/) — 주제 순서, 이론 깊이
- [Andrew Ng ML Specialization (Coursera)](https://www.coursera.org/specializations/machine-learning-introduction) — TF2 Lab 구조
- [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course) — 모듈 구성, 데이터 처리
- 세종대 기계학습개론 (문연국 교수) — 정규 교육과정 연계

---

## 폴더 구조

```
HEART-Lab-Curriculum/
├── README.md                           ← 이 파일
├── ROADMAP.md                          ← 학생용 지도 + 8주 스케줄
├── CHECKPOINTS.md                      ← 관문(GATE)·시험·프로젝트 스펙
├── DIAGNOSTIC_TEST.md                  ← 입구 진단 테스트
├── PAPER_GUIDE.md                      ← 논문 7편 읽기 가이드
├── MANUAL.md                           ← 교육 운영 매뉴얼 (교육자용)
├── CURRICULUM_MASTER.md                ← 장기 설계 문서 (Phase 0~10 비전)
│
├── Chapter01_statistics_data/          ← 통계 + 데이터
│   ├── 1-1_기술통계          .md .py
│   ├── 1-2_확률분포          .md .py
│   ├── 1-3_데이터도구        .md .py
│   ├── 1-4_EDA              .md .py
│   ├── 1-5_결측치_이상치      .md .py
│   └── 1-6_인코딩_스케일링    .md .py
│
├── Chapter02_prediction_classification/ ← 예측 + 분류
│   ├── 2-1_상관분석          .md .py
│   ├── 2-2_선형회귀          .md .py
│   ├── 2-3_손실함수_MSE      .md .py
│   ├── 2-4_경사하강법         .md .py
│   ├── 2-5_로지스틱회귀       .md .py
│   ├── 2-6_손실함수_BCE_CCE  .md .py
│   ├── 2-7_평가지표          .md .py
│   └── 2-8_과적합_일반화      .md .py
│
├── Chapter03_machine_learning/         ← 머신러닝
│   ├── 3-1_결정트리          .md .py
│   ├── 3-2_랜덤포레스트       .md .py
│   ├── 3-3_SVM              .md .py
│   ├── 3-4_KNN              .md .py
│   ├── 3-5_모델비교               .py
│   ├── 3-6_KMeans           .md .py
│   ├── 3-7_PCA              .md .py
│   └── 3-8_ML한계_딥러닝필요성     .py
│
├── Chapter03_5_math_bootcamp/          ← ★ 수학 근육 부트캠프 (손계산)
│   ├── README.md            ← 왜 있는가 + 진행법 (진단은 DIAGNOSTIC_TEST Section A)
│   ├── B-1_미분과_최소화     .md
│   ├── B-2_연쇄법칙_계산그래프 .md
│   ├── B-3_선형대수          .md
│   ├── B-4_확률통계          .md
│   ├── B-5_캡스톤_손역전파_XOR .md .py
│   └── ANSWERS.md           ← 교사용 답안
│
├── Chapter04_deep_learning/            ← 딥러닝 (TF2/Keras)
│   ├── 4-1_뉴런과_신경망     .md
│   ├── 4-2_활성화함수        .md .py
│   ├── 4-3_TF2_첫코드_회귀   .md .py
│   ├── 4-4_회귀_california/diabetes  .py
│   ├── 4-5_이진분류_cancer       .py
│   ├── 4-6_다중분류_iris/mnist   .py
│   ├── 4-7_혼자짜기_5종데이터셋 .md  ← GATE 4 수행 과제
│   └── 4-8_EarlyStopping    .md .py
│
├── worksheets/                         ← 손계산 워크시트 (Ch.5~6 선행)
│   ├── handcalc_CNN.md      ← Conv 출력크기·파라미터 수·MaxPool
│   ├── handcalc_RNN_LSTM.md ← 언롤·기울기 소실·LSTM 게이트
│   └── handcalc_ANSWERS.md  ← 교사용 답안
│
├── _archive/                           ← 이전 버전 (참고용)
└── AI5-main/                           ← 원본 학습 코드
```

---

## 환경 설정 (처음부터)

### 1단계: Anaconda 설치

Anaconda = Python + 데이터 과학 라이브러리가 한번에 설치되는 패키지.

1. https://www.anaconda.com/download 접속
2. 본인 OS에 맞는 버전 다운로드 (Windows/Mac)
3. 설치 시 **"Add to PATH"** 체크
4. 설치 완료 후 **Anaconda Prompt** (Windows) 또는 **터미널** (Mac) 실행

### 2단계: 설치 확인

Anaconda Prompt에서:
```bash
python --version          # Python 3.10+ 확인
conda --version           # conda 설치 확인
```

### 3단계: 추가 패키지 설치

Anaconda에 대부분 포함되어 있지만, 아래 몇 개는 추가 설치 필요:
```bash
pip install scikit-learn        # 머신러닝 (Chapter 2~3)
pip install xgboost             # 앙상블 모델 (Chapter 3)
pip install tensorflow          # 딥러닝 (Chapter 4~)
pip install seaborn             # 시각화 (이미 있을 수 있음)
```

### 4단계: 전부 확인

```python
python -c "
import numpy; print(f'numpy: {numpy.__version__}')
import pandas; print(f'pandas: {pandas.__version__}')
import matplotlib; print(f'matplotlib: {matplotlib.__version__}')
import sklearn; print(f'sklearn: {sklearn.__version__}')
import tensorflow; print(f'tensorflow: {tensorflow.__version__}')
print('All OK!')
"
```

### 자주 쓰는 명령어

| 명령어 | 설명 |
|--------|------|
| `python 파일명.py` | Python 파일 실행 |
| `pip install 패키지명` | 패키지 설치 |
| `pip list` | 설치된 패키지 목록 |
| `conda activate base` | Anaconda 기본 환경 활성화 |
| `conda list` | Anaconda 설치 목록 |

### 코드 에디터 추천

| 에디터 | 특징 |
|--------|------|
| **VS Code** | 가장 추천. 무료, 확장 풍부 |
| **PyCharm** | 파이썬 전용. 학생 무료 |
| **Jupyter Notebook** | 셀 단위 실행. 탐색/시각화에 좋음 |

VS Code 설치 후 **Python 확장** 설치하면 바로 사용 가능.

---

## License

이 커리큘럼은 HEART Lab 내부 교육용으로 제작되었습니다.
