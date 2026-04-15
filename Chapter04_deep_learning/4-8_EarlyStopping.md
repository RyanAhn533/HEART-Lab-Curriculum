# 2-8. 학습을 언제 멈출까 — validation + EarlyStopping + 학습곡선

> **Ref: Coursera C2W3 / AI5 keras15~19**

---

## 왜 배우는가

Phase 1-13에서 **과적합**을 이론으로 배웠다.
이제 실제 코드에서 **과적합을 감지하고 방지**하는 방법을 배운다.

---

## 1. validation_split

```python
model.fit(x_train, y_train, validation_split=0.2)
```

- train 데이터의 20%를 **검증용**으로 떼어놓는다
- 훈련 중에 train loss와 val loss를 동시에 모니터링
- **val loss가 올라가기 시작하면 = 과적합 시작**

---

## 2. EarlyStopping

```python
from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(
    monitor='val_loss',         # 검증 손실을 감시
    patience=10,                # 10번 연속 개선 없으면 멈춤
    restore_best_weights=True,  # 가장 좋았던 가중치로 복원
)

model.fit(x_train, y_train, callbacks=[es], validation_split=0.2)
```

- **patience=10**: val_loss가 10 epoch 연속 개선 안 되면 자동 종료
- **restore_best_weights**: 가장 좋았던 시점의 가중치로 되돌림
- epochs를 크게 잡아도(500) EarlyStopping이 알아서 멈춰줌

---

## 3. 학습곡선으로 과적합 판단

```
train_loss ↓ val_loss ↓  → 아직 학습 중 (더 해도 됨)
train_loss ↓ val_loss ↑  → 과적합! (멈춰야 함)
train_loss ↓ val_loss → → 수렴 (최적점)
```

---

## 나중에 만나는 곳

| 여기서 배운 것 | 나중에 만나는 곳 |
|--------------|---------------|
| validation_split | 모든 Phase의 #3 |
| EarlyStopping | 모든 Phase의 #3 |
| 학습곡선 | Phase 10 ReduceLROnPlateau |

---

## 보고 오기

- **Coursera C2W3**: "Advice for Applying ML"
- **구글 검색**: "keras EarlyStopping 사용법"

---

## 실습

`2-8_EarlyStopping.py` 참조

**★ Phase 2 완료 → 체크포인트 시험 2**
