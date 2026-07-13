# HEART Lab AI 커리큘럼 로드맵 (학생용 지도)

> 통계 → 회귀 → **[수학 부트캠프]** → ANN → CNN → RNN/LSTM → NLP → 전이학습 → Transformer
>
> "일단 돌려보고 → 왜 그런지 파고든다." 영상(직관) → 블로그 리뷰(정리) → 논문(원본)

---

## 왜 부트캠프가 끼어 있나 — 'Chapter 4의 벽'

이 커리큘럼은 의도적으로 top-down이다. 수학부터 시작하지 않고 코드를 돌려본 뒤 이론을 역추적한다.
그 설계의 대가가 Chapter 4에서 나타난다: 순전파·역전파·Chain Rule 같은 '계산 기계'를
3Blue1Brown 영상에 맡기는데, **영상만 보고 손계산을 안 한 사람은 여기서 미끄러진다.**

그래서 Chapter 3와 4 사이에 손계산 부트캠프(`Chapter03_5_math_bootcamp/`)가 있다.
커리큘럼을 바꾸는 게 아니라 비어 있던 레이어 하나를 채우는 다리다.

---

## 전체 흐름

| 단계 | 주제 | 만드는 것 | 핵심 개념 | 필수 영상 | 논문 |
|------|------|----------|----------|----------|------|
| Ch 1 | 통계와 데이터 | 기술통계·확률분포·EDA·전처리 | **표본으로 모집단을 추정한다** | 3B1B "Neural network?" | — |
| Ch 2 | 예측과 분류 | 선형회귀·MSE·경사하강·로지스틱 | 직선 하나로 근사, 기울기 반대로 하강 | 3B1B "Gradient descent" | — |
| Ch 3 | 머신러닝 | 트리·RF·SVM·KNN·PCA·**ML 한계** | 모델을 도구로 비교하는 감각 | StatQuest | — |
| **Ch 3.5** | **수학 부트캠프** | **손계산: 미분→연쇄법칙→선형대수→XOR** | **역전파를 손으로. 계산그래프** | 3B1B "Backprop" Ch.3–4 | — |
| Ch 4 | 딥러닝 입문 (ANN) | TF2 회귀/이진/다중분류 | 회귀는 벽돌, 비선형은 접착제 | 3B1B "NN" Ch.1 | 퍼셉트론 리뷰 |
| Ch 5 | 이미지 (CNN) | Conv2D·MaxPool·증강 | 지역 패턴을 훑는 필터 | 3B1B "Convolution?" | LeNet, AlexNet |
| Ch 6 | 시계열 (RNN/LSTM) | SimpleRNN·LSTM·Bidirectional | 기울기 소실과 게이트 | 3B1B "RNN" | LSTM (1997) |
| Ch 7 | 자연어 (NLP) | Tokenizer·Embedding·감정분류 | 의미를 벡터로 | 3B1B "Word2Vec" | Word2Vec |
| Ch 8 | 성능 올리기 | 튜닝·전이학습(VGG16/ResNet) | 남이 배운 특징을 빌린다 | — | VGG, ResNet |
| Ch 9 | ML 심화 | PCA·KFold·앙상블 | — | 3B1B "Linear Algebra" | XGBoost 리뷰 |
| Ch 10 | PyTorch | Torch 재구현·**Transformer 재현** | Q·K·V, 수동 train loop | 3B1B "Attention" | **Attention Is All You Need** |

체크포인트: Ch 3 끝(시험 1) · Ch 6 끝(시험 2) · Ch 10 끝(최종 발표)

논문은 `PAPER_GUIDE.md` 를 따라 읽는다. 손계산 워크시트는 `worksheets/handcalc_*.md`.

---

## 학생에게 한마디

한 달 만에 이 체인 전체가 안 꿰지는 건 정상이다. 보통 한 학기짜리 흐름이다.
지금 필요한 건 속도가 아니라 **순서**다. 부트캠프의 손계산부터 통과하면,
`model.fit()` 이 마법이 아니라 **네가 손으로 해 본 계산의 자동화**로 보이기 시작한다.
