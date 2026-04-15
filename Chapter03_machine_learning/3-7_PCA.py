# ============================================
# 3-7. PCA (Principal Component Analysis) — Cancer
# 이전(3-6)과 차이: #2에 PCA 추가 (차원 축소)
#
# 왜 배우는가:
#   특성이 너무 많으면 → 중요한 것만 남기고 줄인다.
#   시각화, 학습 속도, 노이즈 제거에 효과적.
#
# ▶ 보고 오기: StatQuest "PCA Main Ideas"
#
# Ref: Stanford CS229 W6 / AI5 ml/m01~05
# ============================================

#0. 라이브러리 ──────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1. 데이터 ─────────────────────────────────
data = load_breast_cancer()
x = data.data        # (569, 30)
y = data.target

print(f"[ Cancer — PCA 차원 축소 ]")
print(f"원본: {x.shape} (30개 특성)")

#2. 데이터 전처리 및 분할 ───────────────────
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()                                           # ← PCA는 스케일링 필수
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# PCA 적용                                                         ← NEW
pca = PCA(n_components=0.95)                                        # 95% 설명 유지
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

n_components = pca.n_components_
print(f"PCA 후: {x_train_pca.shape} ({n_components}개로 축소)")
print(f"설명 비율: {sum(pca.explained_variance_ratio_):.1%}")

#3. 모델 ──────────────────────────────────
# PCA 없이
model_full = RandomForestClassifier(n_estimators=100, random_state=42)
model_full.fit(x_train, y_train)

# PCA 있을 때
model_pca = RandomForestClassifier(n_estimators=100, random_state=42)
model_pca.fit(x_train_pca, y_train)

#4. 평가 ──────────────────────────────────
acc_full = accuracy_score(y_test, model_full.predict(x_test))
acc_pca = accuracy_score(y_test, model_pca.predict(x_test_pca))

print(f"\n[ 평가 — PCA 전후 비교 ]")
print(f"원본 (30D): accuracy={acc_full:.4f}")
print(f"PCA ({n_components}D):  accuracy={acc_pca:.4f}")
print(f"→ {30-n_components}개 특성을 줄여도 성능 {'유지' if abs(acc_full-acc_pca) < 0.02 else '변화'}")

# 누적 설명 비율
pca_full = PCA().fit(x_train)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)
n_95 = np.argmax(cum_var >= 0.95) + 1
print(f"\n[ 누적 설명 비율 ]")
for i in [2, 5, 10, n_95, 30]:
    print(f"  {i:2d}개: {cum_var[i-1]:.1%}")

# 2D 시각화
pca_2d = PCA(n_components=2).fit_transform(x_train)

# ── 시각화 ────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(range(1, 31), cum_var, 'bo-', markersize=4)
axes[0].axhline(0.95, color='red', linestyle='--', label='95%')
axes[0].axvline(n_95, color='green', linestyle='--', label=f'n={n_95}')
axes[0].set_xlabel('Components')
axes[0].set_ylabel('Cumulative Variance')
axes[0].set_title('Scree Plot')
axes[0].legend()

scatter = axes[1].scatter(pca_2d[:, 0], pca_2d[:, 1], c=y_train, cmap='coolwarm', s=20, alpha=0.7)
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
axes[1].set_title('Cancer 2D Visualization')

axes[2].bar(['Original\n(30D)', f'PCA\n({n_components}D)'], [acc_full, acc_pca],
            color=['gray', 'steelblue'], edgecolor='black')
axes[2].set_ylim(0.9, 1.0)
axes[2].set_title('Accuracy: Original vs PCA')

plt.tight_layout()
plt.savefig('3-7_output.png', dpi=100)
plt.show()

print("\n" + "="*50)
print("핵심 정리:")
print(f"  #2에 PCA 추가: 30D → {n_components}D (95% 설명)")
print("  스케일링 필수 (PCA는 분산 기반)")
print("  차원 줄여도 성능 유지 → 노이즈 제거 효과")
print("  2D 시각화로 데이터 구조 파악 가능")
print("="*50)
