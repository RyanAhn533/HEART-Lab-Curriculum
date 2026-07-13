# 논문 읽기 가이드 — 7편, 언제 · 왜 · 어떻게

> **철칙: 영상(직관) → 블로그 "○○ 논문 리뷰"(정리) → 원문(디테일). 절대 논문부터 읽지 마라.**
> 논문은 '공부 자료'가 아니라 '역사적 사건의 기록'이다. 각 논문이 **어떤 문제를 부수러 나왔는지**를 먼저 잡으면 절반은 읽은 것이다.

---

## 읽기 전 준비 (모든 논문 공통)

1. 구글에 "[논문이름] 논문 리뷰" 검색 → 한국어 블로그 2~3개
2. 아래 "한 줄 요약"과 "이 논문이 부순 문제"를 먼저 읽고
3. 원문은 **Abstract → Figure 전부 → Conclusion → (필요하면) 본문** 순서로

읽고 나면 3가지를 적는다 (DIAGNOSTIC_TEST 문제 36~38과 같은 틀):
- Contribution 3줄 요약
- 실험 세팅 (데이터셋 / baseline / 평가지표)
- 한계 2가지 + 개선 아이디어

---

## 1. LeNet-5 — LeCun et al., 1998 (Phase 5 후)

- **한 줄:** Conv + Pooling + FC 라는 CNN의 원형을 만들고 우편번호 손글씨를 읽었다.
- **부순 문제:** 픽셀을 MLP에 그냥 넣으면 파라미터 폭발 + 위치 변화에 취약.
- **볼 것:** Figure 2의 구조도 하나가 논문의 전부다. `handcalc_CNN.md` 문제 2가 이 구조 계산이다.
- **주의:** 옛날 논문이라 길다. 구조도와 아이디어만 챙기고 세부는 버려라.

## 2. AlexNet — Krizhevsky et al., 2012 (Phase 5 후)

- **한 줄:** LeNet을 크게 + ReLU + Dropout + GPU 2장으로 ImageNet을 부수고 딥러닝을 부활시켰다.
- **부순 문제:** "신경망은 깊게 못 만든다"는 통념 (sigmoid 포화, 과적합, 연산량).
- **볼 것:** ReLU가 왜 학습을 빠르게 하나 (Fig.1), Dropout 문단. 전부 Chapter 4~5에서 쓴 것들의 출생신고서다.

## 3. LSTM — Hochreiter & Schmidhuber, 1997 (Phase 6 후)

- **한 줄:** 셀 상태 + 게이트로 RNN의 기울기 소실을 해결해 긴 기억을 가능하게 했다.
- **부순 문제:** `handcalc_RNN_LSTM.md` 문제 2에서 손으로 본 그 소실 (0.5⁹ ≈ 0.002).
- **주의:** **원문이 매우 어렵다.** 이 논문만은 리뷰 블로그 + colah의 "Understanding LSTM Networks" 글로 대체해도 된다. 원문은 아이디어의 존재만 확인.

## 4. Word2Vec — Mikolov et al., 2013 (Phase 7 후)

- **한 줄:** 단어를 주변 단어로 예측하게 학습시켰더니, 의미가 벡터 연산이 됐다 (king − man + woman ≈ queen).
- **부순 문제:** one-hot은 단어 사이의 '의미 거리'가 전부 같다 — 의미를 담을 수 없다.
- **볼 것:** CBOW vs Skip-gram 그림, 벡터 연산 예제 표.

## 5. VGGNet — Simonyan & Zisserman, 2014 (Phase 8)

- **한 줄:** 3×3만 반복해서 깊게 — 단순한 규칙으로 깊이의 힘을 증명했다.
- **부순 문제:** "필터 크기를 어떻게 골라야 하나" → 3×3 두 번 = 5×5 효과인데 파라미터는 더 적다.
- **볼 것:** Table 1 (A~E 구성). 전이학습에서 `VGG16(include_top=False)` 로 매일 만나게 된다.

## 6. ResNet — He et al., 2015 (Phase 8)

- **한 줄:** Skip Connection (x를 더해 주기)으로 152층을 학습 가능하게 했다.
- **부순 문제:** 깊어질수록 오히려 train 에러가 늘어나는 degradation — 과적합이 아니라 **최적화 실패**였다.
- **볼 것:** Fig.1 (20층 vs 56층 역전 그래프), Fig.2 (residual block).
- **연결:** H(x) = F(x) + x 에서 기울기가 +x 경로로 그대로 흐른다 — LSTM의 셀 상태 고속도로와 같은 아이디어다.

## 7. Attention Is All You Need — Vaswani et al., 2017 (Phase 10, 필수)

- **한 줄:** RNN을 버리고 Attention만으로 시퀀스를 처리 — 병렬화 + 긴 의존성, 현대 AI(LLM)의 기반.
- **부순 문제:** RNN은 순차 처리라 느리고, 긴 문장에서 앞 정보가 흐려진다 (Seq2Seq context vector 병목).
- **볼 것:** Q·K·V (질문·색인·내용의 비유), Scaled Dot-Product의 √d_k, Positional Encoding이 필요한 이유.
- **경로:** 3B1B "Attention in transformers" → codingopera.tistory.com/43 → 원문. 이 논문은 Phase 10 재현 프로젝트의 목표물이다.

---

## 전체 지도 (한 눈에)

| # | 논문 | 연도 | 부순 문제 | 시점 |
|---|------|------|----------|------|
| 1 | LeNet-5 | 1998 | 이미지에 MLP는 낭비 | Phase 5 후 |
| 2 | AlexNet | 2012 | "깊게는 못 만든다" | Phase 5 후 |
| 3 | LSTM | 1997 | RNN 기울기 소실 | Phase 6 후 |
| 4 | Word2Vec | 2013 | one-hot엔 의미가 없다 | Phase 7 후 |
| 5 | VGGNet | 2014 | 필터 설계의 복잡함 | Phase 8 |
| 6 | ResNet | 2015 | 깊이의 최적화 실패 | Phase 8 |
| 7 | Attention | 2017 | RNN의 순차 병목 | Phase 10 |

패턴이 보이는가? **모든 논문이 '직전 방법의 병목' 하나를 부순다.**
논문을 읽는다 = 그 병목이 뭐였는지를 아는 것이다. 그래서 순서대로 읽어야 한다.
