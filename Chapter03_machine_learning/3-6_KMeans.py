# ============================================
# 3-6. K-Means (비지도학습 — 군집화)
# 이전(3-5)과 차이: #3에 KMeans (정답 없이 그룹 찾기)
#
# 왜 배우는가:
#   지금까지는 지도학습 (정답 있음).
#   현실에는 정답 없는 데이터도 많다 → 비지도학습.
#
# ▶ 보고 오기: StatQuest "K-Means"
#
# Ref: Coursera C3W1 / Stanford CS229 W6
# ============================================

#0. 라이브러리 ──────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

#1. 데이터 ─────────────────────────────────
x, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)

print(f"[ 합성 데이터 — 4개 군집 ]")
print(f"shape: {x.shape}")

#2. 데이터 전처리 ──────────────────────────
# K-Means는 거리 기반이므로 스케일링 권장
# (make_blobs는 이미 비슷한 범위라 생략 가능)

#3. 모델 ──────────────────────────────────
model = KMeans(n_clusters=4, random_state=42, n_init=10)            # ← NEW: KMeans
model.fit(x)

labels = model.labels_
centers = model.cluster_centers_

#4. 평가 ──────────────────────────────────
ari = adjusted_rand_score(y_true, labels)

print(f"\n[ 평가 ]")
print(f"군집 수: {len(set(labels))}")
print(f"Adjusted Rand Index: {ari:.4f} (1이면 완벽)")
print(f"Inertia: {model.inertia_:.1f}")

# Elbow Method
print(f"\n[ Elbow Method — K 선택 ]")
inertias = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(x)
    inertias.append(km.inertia_)
    print(f"  K={k:2d}: Inertia={km.inertia_:8.1f}")

# Iris에도 적용
iris = load_iris()
x_iris = StandardScaler().fit_transform(iris.data)
km_iris = KMeans(n_clusters=3, random_state=42, n_init=10).fit(x_iris)
ari_iris = adjusted_rand_score(iris.target, km_iris.labels_)
print(f"\n[ Iris K-Means: ARI={ari_iris:.4f} ]")

# ── 시각화 ────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].scatter(x[:, 0], x[:, 1], c=labels, cmap='viridis', s=20)
axes[0].scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, edgecolors='black')
axes[0].set_title(f'K-Means (K=4, ARI={ari:.3f})')

axes[1].plot(range(1, 11), inertias, 'bo-', markersize=6)
axes[1].axvline(4, color='red', linestyle='--', alpha=0.5)
axes[1].set_xlabel('K')
axes[1].set_ylabel('Inertia')
axes[1].set_title('Elbow Method')

axes[2].scatter(x_iris[:, 0], x_iris[:, 1], c=km_iris.labels_, cmap='viridis', s=20)
axes[2].set_title(f'Iris K-Means (ARI={ari_iris:.3f})')

plt.tight_layout()
plt.savefig('3-6_output.png', dpi=100)
plt.show()

print("\n" + "="*50)
print("핵심 정리:")
print("  #3만 바뀜: KMeans(n_clusters=4)")
print("  비지도학습 = 정답(y) 없이 패턴 발견")
print("  Elbow Method = K를 선택하는 방법")
print("  ARI = 군집 결과와 실제 라벨 비교")
print("="*50)
