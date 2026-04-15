# ============================================
# 3-2. 랜덤포레스트 (Random Forest) — Cancer
# 이전(3-1)과 차이: #3에 RandomForestClassifier (트리 여러 개)
#
# 왜 배우는가:
#   트리 1개는 불안정 → 100개 합치면 안정적.
#   테이블 데이터에서 가장 강력한 모델 중 하나.
#
# ▶ 보고 오기: StatQuest "Random Forests"
#
# Ref: Coursera C2W4 Lab02 / AI5 ml/m37
# ============================================

#0. 라이브러리 ──────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

#1. 데이터 ─────────────────────────────────
data = load_breast_cancer()
x = data.data        # (569, 30)
y = data.target       # (569,)

print(f"[ Breast Cancer — 랜덤포레스트 ]")
print(f"shape: {x.shape}")

#2. 데이터 전처리 및 분할 ───────────────────
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
# 트리 계열은 스케일링 불필요

#3. 모델 ──────────────────────────────────
model = RandomForestClassifier(                      # ← NEW: 트리 100개 앙상블
    n_estimators=100,
    max_depth=5,
    random_state=42,
)
model.fit(x_train, y_train)

# 비교용: 단일 결정트리
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(x_train, y_train)

#4. 평가 ──────────────────────────────────
y_pred = model.predict(x_test)
acc_rf = accuracy_score(y_test, y_pred)
acc_dt = accuracy_score(y_test, dt.predict(x_test))

print(f"\n[ 평가 ]")
print(f"Decision Tree:   {acc_dt:.4f}")
print(f"Random Forest:   {acc_rf:.4f}")
print(f"→ RF가 {'더 좋음' if acc_rf > acc_dt else '비슷'}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Malignant', 'Benign'])}")

# 특성 중요도 (상위 10개)
importance = model.feature_importances_
top_idx = np.argsort(importance)[-10:]
print("[ 특성 중요도 상위 10 ]")
for idx in reversed(top_idx):
    bar = "█" * int(importance[idx] * 50)
    print(f"  {data.feature_names[idx]:25s}: {importance[idx]:.3f} {bar}")

# n_estimators별 성능
print(f"\n[ 트리 개수별 성능 ]")
for n in [1, 10, 50, 100, 200]:
    rf = RandomForestClassifier(n_estimators=n, random_state=42).fit(x_train, y_train)
    print(f"  n={n:4d}: {accuracy_score(y_test, rf.predict(x_test)):.4f}")

# XGBoost (있으면)
try:
    from xgboost import XGBClassifier
    xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                         random_state=42, verbosity=0)
    xgb.fit(x_train, y_train)
    acc_xgb = accuracy_score(y_test, xgb.predict(x_test))
    print(f"\n[ XGBoost: {acc_xgb:.4f} ]")
except ImportError:
    print(f"\n[ XGBoost 미설치 — pip install xgboost ]")

# ── 시각화 ────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].bar(['Decision Tree', 'Random Forest'], [acc_dt, acc_rf],
            color=['salmon', 'steelblue'], edgecolor='black')
axes[0].set_title('DT vs RF')
axes[0].set_ylim(0.9, 1.0)

axes[1].barh(np.array(data.feature_names)[top_idx], importance[top_idx], color='steelblue')
axes[1].set_title('Top 10 Feature Importance')

single_accs = [accuracy_score(y_test,
    DecisionTreeClassifier(max_depth=5, random_state=i).fit(x_train, y_train).predict(x_test))
    for i in range(50)]
axes[2].hist(single_accs, bins=15, alpha=0.7, edgecolor='black', label='Single Trees')
axes[2].axvline(acc_rf, color='red', linewidth=2, linestyle='--', label=f'RF={acc_rf:.4f}')
axes[2].set_title('Single Trees vs RF')
axes[2].legend()

plt.tight_layout()
plt.savefig('3-2_output.png', dpi=100)
plt.show()

# ── 정리 ──────────────────────────────────
print("\n" + "="*50)
print("핵심 정리:")
print("  #3만 바뀜: RandomForestClassifier(n_estimators=100)")
print("  = 트리 100개 + 투표 (Bagging + 특성 랜덤)")
print("  n_estimators ↑ → 성능 ↑ (100이면 충분)")
print("  테이블 데이터 최강 → XGBoost도 같은 계열")
print("="*50)
