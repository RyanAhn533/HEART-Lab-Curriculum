# ============================================
# 1-10. 로지스틱 회귀 (Logistic Regression)
#
# 왜 배우는가:
#   분류 문제의 시작. 선형회귀 + Sigmoid = 분류 모델.
#   모든 신경망 이진분류의 기초.
#
# 나중에 만나는 곳:
#   → Phase 5~6: activation='sigmoid' / 'softmax'
#   → 1-11: BCE (이진분류 손실함수)
#
# ▶ 보고 오기: Coursera C1W3 "Logistic Regression"
#
# Ref: Stanford CS229 W2 / Google MLCC
# ============================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ── 1. Sigmoid 함수 이해 ─────────────────
print("[ Sigmoid 함수 ]")

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z_values = np.linspace(-10, 10, 100)
sig_values = sigmoid(z_values)

print(f"z=-10 → sigmoid={sigmoid(-10):.6f} (거의 0)")
print(f"z=  0 → sigmoid={sigmoid(0):.1f}    (정확히 0.5)")
print(f"z=+10 → sigmoid={sigmoid(10):.6f} (거의 1)")

# ── 2. 이진분류 (Breast Cancer) ──────────
print(f"\n[ 이진분류 — Breast Cancer ]")
data = load_breast_cancer()
X = data.data
y = data.target  # 0=악성, 1=양성

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # 양성 확률

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
print(f"예측 확률 (처음 5개): {y_prob[:5].round(3)}")
print(f"예측 클래스: {y_pred[:5]}")
print(f"실제 클래스: {y_test[:5]}")

# ── 3. 다중분류 (Iris) ───────────────────
print(f"\n[ 다중분류 — Iris (3 클래스) ]")
iris = load_iris()
X_iris = iris.data[:, :2]  # 시각화를 위해 2개 특성만
y_iris = iris.target

X_tr, X_te, y_tr, y_te = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)
scaler2 = StandardScaler()
X_tr = scaler2.fit_transform(X_tr)
X_te = scaler2.transform(X_te)

model_multi = LogisticRegression(max_iter=1000)
model_multi.fit(X_tr, y_tr)
acc_multi = model_multi.score(X_te, y_te)
print(f"Accuracy: {acc_multi:.4f}")

# ── 4. 시각화 ─────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# Sigmoid 함수
axes[0, 0].plot(z_values, sig_values, 'b-', linewidth=2)
axes[0, 0].axhline(0.5, color='red', linestyle='--', alpha=0.5, label='threshold=0.5')
axes[0, 0].axvline(0, color='gray', linestyle=':', alpha=0.5)
axes[0, 0].fill_between(z_values, sig_values, 0.5, where=(sig_values >= 0.5),
                         alpha=0.2, color='green', label='Positive')
axes[0, 0].fill_between(z_values, sig_values, 0.5, where=(sig_values < 0.5),
                         alpha=0.2, color='red', label='Negative')
axes[0, 0].set_title('Sigmoid Function')
axes[0, 0].set_xlabel('z')
axes[0, 0].set_ylabel('σ(z)')
axes[0, 0].legend(fontsize=8)

# 예측 확률 분포
axes[0, 1].hist(y_prob[y_test == 1], bins=15, alpha=0.7, label='Positive', color='green')
axes[0, 1].hist(y_prob[y_test == 0], bins=15, alpha=0.7, label='Negative', color='red')
axes[0, 1].axvline(0.5, color='black', linestyle='--', label='Threshold')
axes[0, 1].set_title('Predicted Probabilities')
axes[0, 1].set_xlabel('Probability')
axes[0, 1].legend()

# 실제 vs 예측
axes[0, 2].scatter(range(len(y_test)), y_test, color='blue', s=30, label='Actual', alpha=0.7)
axes[0, 2].scatter(range(len(y_pred)), y_pred, color='red', s=10, label='Predicted', alpha=0.7)
axes[0, 2].set_title(f'Actual vs Predicted (Acc={acc:.3f})')
axes[0, 2].set_ylabel('Class')
axes[0, 2].legend()

# 결정경계 (Iris 2D)
h = 0.02
x_min, x_max = X_tr[:, 0].min() - 1, X_tr[:, 0].max() + 1
y_min, y_max = X_tr[:, 1].min() - 1, X_tr[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = model_multi.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
axes[1, 0].contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
scatter = axes[1, 0].scatter(X_tr[:, 0], X_tr[:, 1], c=y_tr, cmap='viridis', s=30, edgecolors='k')
axes[1, 0].set_title('Decision Boundary (Iris)')
axes[1, 0].set_xlabel('Feature 1')
axes[1, 0].set_ylabel('Feature 2')

# Softmax 시각화
def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / exp_z.sum()

logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)
axes[1, 1].bar(['Class 0', 'Class 1', 'Class 2'], probs,
               color=['steelblue', 'green', 'orange'], edgecolor='black')
axes[1, 1].set_title(f'Softmax Output\n{probs.round(3)}')
axes[1, 1].set_ylabel('Probability')

# 선형회귀 vs 로지스틱 비교
x_demo = np.linspace(-3, 3, 100)
axes[1, 2].plot(x_demo, x_demo, 'b--', label='Linear (unbounded)', alpha=0.7)
axes[1, 2].plot(x_demo, sigmoid(x_demo), 'r-', linewidth=2, label='Logistic (0~1)')
axes[1, 2].set_title('Linear vs Logistic')
axes[1, 2].legend()
axes[1, 2].set_xlabel('Input')
axes[1, 2].set_ylabel('Output')

plt.tight_layout()
plt.savefig('1-10_output.png', dpi=100)
plt.show()

# ── 정리 ──────────────────────────────────
print("\n" + "="*50)
print("핵심 정리:")
print("  로지스틱 회귀 = 선형회귀 + Sigmoid")
print("  Sigmoid → 출력을 0~1 확률로 압축")
print("  threshold 0.5 → 이진분류")
print("  Softmax → 다중분류 (확률 합 = 1)")
print("  → 이제 이 분류를 '평가'하는 방법이 필요 (1-12)")
print("="*50)
