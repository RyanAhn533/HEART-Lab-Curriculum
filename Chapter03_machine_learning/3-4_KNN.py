# ============================================
# 3-4. KNN (K-Nearest Neighbors) — Iris
# 이전(3-3)과 차이: #3에 KNeighborsClassifier
#
# 왜 배우는가:
#   가장 직관적인 분류. 가까운 K개의 다수결.
#
# ▶ 보고 오기: StatQuest "KNN"
# ============================================

#0. 라이브러리 ──────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler                    # ← KNN도 스케일링 필수
from sklearn.metrics import accuracy_score, classification_report

#1. 데이터 ─────────────────────────────────
data = load_iris()
x = data.data        # (150, 4)
y = data.target

print(f"[ Iris — KNN ]")
print(f"shape: {x.shape}")

#2. 데이터 전처리 및 분할 ───────────────────
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()                                           # ← KNN은 거리 기반이라 스케일링 필수
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#3. 모델 ──────────────────────────────────
model = KNeighborsClassifier(n_neighbors=5)                         # ← NEW: KNN
model.fit(x_train, y_train)

#4. 평가 ──────────────────────────────────
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n[ 평가 ]")
print(f"Accuracy: {acc:.4f} (K=5)")
print(f"\n{classification_report(y_test, y_pred, target_names=data.target_names)}")

# K값별 성능
print("[ K값별 성능 ]")
k_range = range(1, 21)
k_accs = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k).fit(x_train, y_train)
    k_accs.append(accuracy_score(y_test, knn.predict(x_test)))
    if k <= 10:
        print(f"  K={k:2d}: {k_accs[-1]:.4f}")

best_k = list(k_range)[np.argmax(k_accs)]
print(f"\n최적 K: {best_k} (acc={max(k_accs):.4f})")

# ── 시각화 ────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(list(k_range), k_accs, 'b-o', markersize=5)
axes[0].axvline(best_k, color='red', linestyle='--', label=f'Best K={best_k}')
axes[0].set_xlabel('K')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('K vs Accuracy')
axes[0].legend()

def plot_knn(ax, k, X, y, title):
    knn = KNeighborsClassifier(n_neighbors=k).fit(X, y)
    h = 0.02
    xx, yy = np.meshgrid(np.arange(X[:,0].min()-0.5, X[:,0].max()+0.5, h),
                          np.arange(X[:,1].min()-0.5, X[:,1].max()+0.5, h))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=20, edgecolors='k')
    ax.set_title(title)

x_2d = x_train[:, :2]
plot_knn(axes[1], 1, x_2d, y_train, 'K=1 (Overfitting)')
plot_knn(axes[2], best_k, x_2d, y_train, f'K={best_k} (Good)')

plt.tight_layout()
plt.savefig('3-4_output.png', dpi=100)
plt.show()

# ── 정리 ──────────────────────────────────
print("\n" + "="*50)
print("핵심 정리:")
print(f"  #3만 바뀜: KNeighborsClassifier(n_neighbors={best_k})")
print("  #2에 StandardScaler (거리 기반이라 필수)")
print("  K=1 과적합, K 크면 과소적합")
print("  학습 없음 → 데이터 많으면 느림")
print("="*50)
