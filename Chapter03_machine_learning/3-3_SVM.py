# ============================================
# 3-3. SVM (Support Vector Machine) — Cancer
# 이전(3-2)과 차이: #3에 SVC + #2에 StandardScaler 필수
#
# 왜 배우는가:
#   "가장 여유 있게 나누는 선" = 마진 최대화.
#   고차원 데이터에서 강력. 커널로 비선형도 처리.
#
# ▶ 보고 오기: StatQuest "SVM"
#
# Ref: Stanford CS229 W3 / 정규과정 SVM 슬라이드
# ============================================

#0. 라이브러리 ──────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler                    # ← SVM은 스케일링 필수!
from sklearn.metrics import accuracy_score, classification_report

#1. 데이터 ─────────────────────────────────
data = load_breast_cancer()
x = data.data        # (569, 30)
y = data.target

print(f"[ Breast Cancer — SVM ]")
print(f"shape: {x.shape}")

#2. 데이터 전처리 및 분할 ───────────────────
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()                                           # ← NEW: SVM은 스케일링 필수
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#3. 모델 ──────────────────────────────────
model = SVC(kernel='rbf', C=1.0)                                    # ← NEW: SVC
model.fit(x_train, y_train)

#4. 평가 ──────────────────────────────────
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n[ 평가 ]")
print(f"Accuracy: {acc:.4f}")
print(f"서포트 벡터 수: {model.n_support_}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Malignant', 'Benign'])}")

# ── 추가: 커널 비교 (비선형 데이터) ──────
print("[ 커널 비교 — Moon 데이터 (비선형) ]")
x_moon, y_moon = make_moons(n_samples=300, noise=0.2, random_state=42)
xm_tr, xm_te, ym_tr, ym_te = train_test_split(x_moon, y_moon, test_size=0.2, random_state=42)

for k in ['linear', 'rbf', 'poly']:
    svm = SVC(kernel=k, C=1.0).fit(xm_tr, ym_tr)
    acc_k = accuracy_score(ym_te, svm.predict(xm_te))
    print(f"  kernel={k:8s}: {acc_k:.4f}")

# ── 시각화 ────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

def plot_boundary(ax, model, X, y, title):
    h = 0.02
    xx, yy = np.meshgrid(np.arange(X[:,0].min()-0.5, X[:,0].max()+0.5, h),
                          np.arange(X[:,1].min()-0.5, X[:,1].max()+0.5, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=20, edgecolors='k')
    ax.set_title(title)

for i, k in enumerate(['linear', 'rbf', 'poly']):
    svm = SVC(kernel=k, C=1.0).fit(xm_tr, ym_tr)
    acc_k = accuracy_score(ym_te, svm.predict(xm_te))
    plot_boundary(axes[i], svm, xm_tr, ym_tr, f'{k} (acc={acc_k:.3f})')

plt.tight_layout()
plt.savefig('3-3_output.png', dpi=100)
plt.show()

# ── 정리 ──────────────────────────────────
print("\n" + "="*50)
print("핵심 정리:")
print("  #3만 바뀜: SVC(kernel='rbf', C=1.0)")
print("  #2에 StandardScaler 추가 (SVM은 스케일링 필수!)")
print("  kernel='rbf' → 대부분 잘 작동")
print("  C 크면 과적합, 작으면 과소적합")
print("="*50)
