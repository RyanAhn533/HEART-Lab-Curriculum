# ============================================
# 1-12. 평가지표 (Evaluation Metrics)
#
# 왜 배우는가:
#   "모델이 잘 했는가?"를 판단하는 기준.
#   Accuracy만 보면 큰 실수를 할 수 있다.
#
# 나중에 만나는 곳:
#   → Phase 5~6: 분류 모델 평가
#   → 논문 결과표: Accuracy, F1, AUC
#
# ▶ 보고 오기: StatQuest "ROC and AUC"
#
# Ref: Google MLCC "Classification" / Coursera C2W3
# ============================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, classification_report,
                              accuracy_score, precision_score, recall_score,
                              f1_score, roc_curve, auc)
import seaborn as sns

# ── 1. 모델 학습 ─────────────────────────
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ── 2. 혼동행렬 ──────────────────────────
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("[ 혼동행렬 ]")
print(f"  TP={tp} (양성을 양성으로)")
print(f"  FP={fp} (음성을 양성으로 — 1종 오류)")
print(f"  FN={fn} (양성을 음성으로 — 2종 오류)")
print(f"  TN={tn} (음성을 음성으로)")

# ── 3. 지표 계산 ─────────────────────────
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n[ 평가지표 ]")
print(f"  Accuracy:  {acc:.4f}  (전체 중 맞춘 비율)")
print(f"  Precision: {prec:.4f}  (양성 예측 중 진짜 양성)")
print(f"  Recall:    {rec:.4f}  (진짜 양성 중 찾은 비율)")
print(f"  F1 Score:  {f1:.4f}  (Precision과 Recall의 균형)")

print(f"\n[ classification_report — 실무에서 이걸 쓴다 ]")
print(classification_report(y_test, y_pred, target_names=['Malignant', 'Benign']))

# ── 4. ROC / AUC ─────────────────────────
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
print(f"[ ROC AUC = {roc_auc:.4f} ]")

# ── 5. Accuracy의 함정 시연 ──────────────
print(f"\n[ Accuracy의 함정 — 불균형 데이터 ]")
y_imbalanced = np.array([0]*990 + [1]*10)  # 1%만 양성
y_all_negative = np.zeros(1000)             # 전부 음성이라고 예측
acc_trap = accuracy_score(y_imbalanced, y_all_negative)
rec_trap = recall_score(y_imbalanced, y_all_negative, zero_division=0)
print(f"  전부 음성으로 예측:")
print(f"  Accuracy = {acc_trap:.1%} ← 높아 보이지만")
print(f"  Recall   = {rec_trap:.1%} ← 양성을 하나도 못 찾음!")

# ── 6. 시각화 ─────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# 혼동행렬 히트맵
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
            xticklabels=['Malignant', 'Benign'], yticklabels=['Malignant', 'Benign'])
axes[0, 0].set_title('Confusion Matrix')
axes[0, 0].set_xlabel('Predicted')
axes[0, 0].set_ylabel('Actual')

# 지표 바 차트
metrics = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1}
axes[0, 1].bar(metrics.keys(), metrics.values(), color='steelblue', edgecolor='black')
axes[0, 1].set_title('Classification Metrics')
axes[0, 1].set_ylim(0, 1)
for i, (k, v) in enumerate(metrics.items()):
    axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)

# ROC 곡선
axes[0, 2].plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {roc_auc:.3f}')
axes[0, 2].plot([0, 1], [0, 1], 'r--', label='Random (AUC=0.5)')
axes[0, 2].fill_between(fpr, tpr, alpha=0.1)
axes[0, 2].set_title('ROC Curve')
axes[0, 2].set_xlabel('False Positive Rate')
axes[0, 2].set_ylabel('True Positive Rate')
axes[0, 2].legend()

# Threshold에 따른 Precision/Recall 변화
precisions = []
recalls = []
thres_range = np.arange(0.1, 1.0, 0.05)
for t in thres_range:
    y_t = (y_prob >= t).astype(int)
    precisions.append(precision_score(y_test, y_t, zero_division=0))
    recalls.append(recall_score(y_test, y_t, zero_division=0))
axes[1, 0].plot(thres_range, precisions, 'b-', label='Precision')
axes[1, 0].plot(thres_range, recalls, 'r-', label='Recall')
axes[1, 0].axvline(0.5, color='gray', linestyle='--', label='default=0.5')
axes[1, 0].set_title('Precision/Recall vs Threshold')
axes[1, 0].set_xlabel('Threshold')
axes[1, 0].legend()

# Accuracy 함정
axes[1, 1].bar(['Accuracy\n(misleading)', 'Recall\n(truth)'],
               [acc_trap, rec_trap],
               color=['orange', 'red'], edgecolor='black')
axes[1, 1].set_title('Accuracy Trap (Imbalanced)')
axes[1, 1].set_ylim(0, 1.1)

# 예측 확률 분포
axes[1, 2].hist(y_prob[y_test == 1], bins=15, alpha=0.7, label='Positive', color='green')
axes[1, 2].hist(y_prob[y_test == 0], bins=15, alpha=0.7, label='Negative', color='red')
axes[1, 2].axvline(0.5, color='black', linestyle='--')
axes[1, 2].set_title('Predicted Probability Distribution')
axes[1, 2].legend()

plt.tight_layout()
plt.savefig('1-12_output.png', dpi=100)
plt.show()

# ── 정리 ──────────────────────────────────
print("\n" + "="*50)
print("핵심 정리:")
print("  혼동행렬 → TP, FP, FN, TN 한눈에")
print("  Accuracy → 불균형 데이터에서 위험")
print("  Precision → FP 줄이기 (스팸필터)")
print("  Recall → FN 줄이기 (암 진단)")
print("  F1 → 둘 다 중요할 때")
print("  AUC → 전체적인 분류 성능")
print("="*50)
