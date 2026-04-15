# 1-11. 분류의 오차 측정 — BCE / CCE

> **Ref: Coursera C1W3 비용함수 / Google MLCC "Logistic Regression"**

---

## 왜 배우는가

1-8에서 회귀용 손실함수(MSE)를 배웠다. 분류 문제에서는 MSE가 잘 안 된다.
분류에 맞는 손실함수가 **Binary Cross-Entropy(BCE)**와 **Categorical Cross-Entropy(CCE)**.

---

## 1. 왜 분류에 MSE를 안 쓰는가?

Sigmoid 출력에 MSE를 적용하면 **손실 곡면이 울퉁불퉁**해서
경사하강법이 최저점을 못 찾는다 (local minimum에 빠짐).

BCE를 쓰면 손실 곡면이 **매끄러운 볼록 함수**가 되어 최적화가 잘 된다.

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

## 나중에 만나는 곳

| 여기서 배운 것 | 나중에 만나는 곳 | 어떻게 쓰이는가 |
|--------------|---------------|---------------|
| BCE | Phase 5~6 이진분류 | `loss='binary_crossentropy'` |
| CCE | Phase 5~6 다중분류 | `loss='categorical_crossentropy'` |
| 문제→손실 매핑 | 모든 Phase의 #3 | 문제 유형 보고 loss 선택 |

---

## 보고 오기

- **Coursera C1W3**: "Cost Function for Logistic Regression"
- **구글 검색**: "cross entropy 직관적 이해"
- **StatQuest**: "Cross Entropy"

---

## 실습

`1-11_손실함수_BCE_CCE.py` 참조

*다음: 1-12. 모델이 잘 했는가 — 평가지표*
