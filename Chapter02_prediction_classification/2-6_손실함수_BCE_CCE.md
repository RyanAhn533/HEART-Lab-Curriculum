# 2-6. 분류의 오차 측정 — BCE / CCE

> **Ref: Coursera C1W3 비용함수 / Google MLCC "Logistic Regression"**

---

## 왜 배우는가

2-3에서 회귀용 손실함수(MSE)를 배웠다. 분류 문제에서는 MSE가 잘 안 된다.
분류에 맞는 손실함수가 **Binary Cross-Entropy(BCE)**와 **Categorical Cross-Entropy(CCE)**.

---

## 1. 왜 분류에 MSE를 안 쓰는가?

Sigmoid 출력에 MSE를 결합하면 손실 함수가 **비볼록(non-convex)**해지고,
sigmoid가 0이나 1에 가까운 **포화 구간에서는 기울기가 거의 0**이라
경사하강법의 학습이 매우 느려진다. **핵심 문제는 기울기 소실**이다.

BCE를 쓰면 손실이 **볼록(convex)**해지고, 확신하며 틀릴수록 기울기가 커져서 학습이 잘 된다.

---

## 2. Binary Cross-Entropy (BCE)

이진분류 (0 또는 1)용 손실함수.

```
BCE = -(1/n) × Σ [y×log(p) + (1-y)×log(1-p)]

y: 실제값 (0 or 1)
p: 예측 확률 (0~1)
```

### 직관

| 실제 | 예측 | 손실 | 설명 |
|------|------|------|------|
| 1 | 0.99 | 0.01 | 잘 맞춤 → 낮은 손실 |
| 1 | 0.01 | 4.61 | 완전 틀림 → 높은 손실 |
| 0 | 0.01 | 0.01 | 잘 맞춤 → 낮은 손실 |
| 0 | 0.99 | 4.61 | 완전 틀림 → 높은 손실 |

**"확신하면서 틀리면 페널티가 극대화된다"**

---

## 3. Categorical Cross-Entropy (CCE)

다중분류 (3개 이상 클래스)용 손실함수.

```
CCE = -Σ y_i × log(p_i)

y_i: One-Hot 인코딩된 실제값 (해당 클래스만 1)
p_i: Softmax 출력 (각 클래스 확률)
```

---

## 4. 정리: 문제 유형별 손실함수

| 문제 | 출력 활성화 | 손실함수 | 코드 |
|------|-----------|---------|------|
| 회귀 | 없음 (linear) | MSE | `loss='mse'` |
| 이진분류 | Sigmoid | BCE | `loss='binary_crossentropy'` |
| 다중분류 | Softmax | CCE | `loss='categorical_crossentropy'` |

---

## 5. 고리 닫기 — 2-4 + 2-6 = 2-5

경사하강법(2-4)으로 BCE(2-6)를 최소화하면, 그게 바로 **로지스틱 회귀(2-5)**다.

```
z = wx + b  →  p = sigmoid(z)  →  BCE(y, p)  →  GD로 w, b 업데이트 → 반복
```

sklearn의 `LogisticRegression()`이 내부에서 하는 일이 정확히 이것.
실습 py 말미에 numpy 20줄로 직접 구현해서 sklearn 결과와 비교한다.

---

## 나중에 만나는 곳

| 여기서 배운 것 | 나중에 만나는 곳 | 어떻게 쓰이는가 |
|--------------|---------------|---------------|
| BCE | Chapter 4 이진분류 | `loss='binary_crossentropy'` |
| CCE | Chapter 4 다중분류 | `loss='categorical_crossentropy'` |
| 문제→손실 매핑 | 모든 Chapter의 #3 | 문제 유형 보고 loss 선택 |

---

## 보고 오기

- **Coursera C1W3**: "Cost Function for Logistic Regression"
- **구글 검색**: "cross entropy 직관적 이해"
- **StatQuest**: "Cross Entropy"

---

## 실습

`2-6_손실함수_BCE_CCE.py` 참조

*다음: 2-7. 모델이 잘 했는가 — 평가지표*
