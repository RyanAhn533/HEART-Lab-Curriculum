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
Chapter 1: 통계와 데이터           ─── "데이터가 어떻게 생겼는지 본다"
Chapter 2: 예측과 분류의 기초       ─── "직선으로 예측하고, 분류한다"
Chapter 3: 머신러닝 모델들          ─── "다양한 모델을 배우고 비교한다"
Chapter 4: 딥러닝 입문 (TF2/Keras) ─── "신경망으로 확장한다"
                                   (Chapter 5~10 개발 중)
```

| Chapter | 주제 | 항목 수 | 프레임워크 |
|---------|------|--------|-----------|
| **1. 통계와 데이터** | 기술통계, 확률분포, EDA, 전처리 | 5개 | numpy, pandas |
| **2. 예측과 분류** | 선형회귀, 경사하강법, 로지스틱, 평가지표 | 8개 | sklearn |
| **3. 머신러닝** | 결정트리, RF, SVM, KNN, K-Means, PCA | 8개 | sklearn |
| **4. 딥러닝** | 뉴런, 활성화, TF2 회귀/분류, EarlyStopping | 8개 | TF2/Keras |

---

## 7주 진행 계획

| 주차 | 내용 | Chapter |
|------|------|---------|
| **Week 1** | 통계 기초 + 데이터 탐색/전처리 | Ch.1 (1-1 ~ 1-5) |
| **Week 2** | 상관분석 + 선형회귀 + 손실함수 + 경사하강법 | Ch.2 전반 (2-1 ~ 2-4) |
| **Week 3** | 로지스틱 회귀 + 평가지표 + 과적합 | Ch.2 후반 (2-5 ~ 2-8) |
| **Week 4** | 결정트리 + RF + SVM + KNN + 모델 비교 | Ch.3 전반 (3-1 ~ 3-5) |
| **Week 5** | K-Means + PCA + ML 한계 → 딥러닝 이론 | Ch.3 후반 + Ch.4 이론 |
| **Week 6** | TF2 실습 (회귀/이진분류/다중분류/EarlyStopping) | Ch.4 실습 (4-3 ~ 4-8) |
| **Week 7** | 종합 복습 + 자유 프로젝트 + 최종 발표 | 종합 |

---

## 대상별 시작점

| 대상 | 시작 | 예상 기간 |
|------|------|----------|
| 학부 3학년 (코딩 초보) | Chapter 1부터 | 7주 |
| 타 도메인 석박 (통계O, 코딩X) | Chapter 1 빠르게 → Chapter 2 | 5~6주 |
| CS 석사 (ML 경험O) | Chapter 3 or 4 | 3~4주 |

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
├── CURRICULUM_MASTER.md                ← 전체 설계 문서
├── DIAGNOSTIC_TEST.md                  ← 진단 테스트
├── MANUAL.md                           ← 교육 운영 매뉴얼
│
├── Chapter01_statistics_data/          ← 통계 + 데이터
│   ├── 1-1_기술통계          .md .py
│   ├── 1-2_확률분포          .md .py
│   ├── 1-3_EDA              .md .py
│   ├── 1-4_결측치_이상치      .md .py
│   └── 1-5_인코딩_스케일링    .md .py
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
├── Chapter04_deep_learning/            ← 딥러닝 (TF2/Keras)
│   ├── 4-1_뉴런과_신경망     .md
│   ├── 4-2_활성화함수        .md .py
│   ├── 4-3_TF2_첫코드_회귀   .md .py
│   ├── 4-4_회귀_boston/diabetes  .py
│   ├── 4-5_이진분류_cancer       .py
│   ├── 4-6_다중분류_iris/mnist   .py
│   └── 4-8_EarlyStopping    .md .py
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
