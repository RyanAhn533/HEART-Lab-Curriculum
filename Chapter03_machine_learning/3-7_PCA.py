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
import numpy as np                              # numpy: 숫자 배열(행렬) 연산 라이브러리
import matplotlib.pyplot as plt                  # matplotlib.pyplot: 그래프/차트 시각화 라이브러리
from sklearn.decomposition import PCA            # PCA: 주성분 분석 — 데이터의 분산(정보)을 가장 많이 보존하는 방향으로 차원을 줄이는 기법
from sklearn.datasets import load_breast_cancer  # load_breast_cancer: 유방암 데이터 (30개 특성)
from sklearn.preprocessing import StandardScaler # StandardScaler: 정규화 — PCA는 분산 기반이라 스케일링 필수!
from sklearn.ensemble import RandomForestClassifier  # RandomForestClassifier: 랜덤포레스트 (PCA 전후 성능 비교용)
from sklearn.model_selection import train_test_split  # train_test_split: 데이터 분할
from sklearn.metrics import accuracy_score       # accuracy_score: 정확도 계산

#1. 데이터 ─────────────────────────────────
data = load_breast_cancer()   # 유방암 데이터 로드 (569명, 30개 특성)
x = data.data                 # x: 입력 특성 (569, 30)
y = data.target               # y: 정답 (0=악성, 1=양성)

print(f"[ Cancer — PCA 차원 축소 ]")
print(f"원본: {x.shape} (30개 특성)")

#2. 데이터 전처리 및 분할 ───────────────────
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42    # 20% 테스트
)
scaler = StandardScaler()                    # PCA 전에 스케일링 필수 (특성 단위가 다르면 분산이 왜곡됨)
x_train = scaler.fit_transform(x_train)      # 훈련 데이터 기준 정규화
x_test = scaler.transform(x_test)            # 같은 기준으로 테스트 데이터 변환

# PCA 적용                                                         ← NEW
pca = PCA(n_components=0.95)                  # n_components=0.95: 전체 분산(정보)의 95%를 보존하는 최소 개수의 주성분만 남김
x_train_pca = pca.fit_transform(x_train)      # fit_transform: 주성분 방향 계산(fit) + 데이터를 그 방향으로 변환(transform)
x_test_pca = pca.transform(x_test)            # transform: 같은 주성분 방향으로 테스트 데이터도 변환

n_components = pca.n_components_              # n_components_: 실제로 선택된 주성분 개수
print(f"PCA 후: {x_train_pca.shape} ({n_components}개로 축소)")
print(f"설명 비율: {sum(pca.explained_variance_ratio_):.1%}")  # explained_variance_ratio_: 각 주성분이 전체 분산 중 몇 %를 설명하는지

#3. 모델 ──────────────────────────────────
# PCA 없이 (원본 30개 특성 그대로)
model_full = RandomForestClassifier(n_estimators=100, random_state=42)  # 트리 100개
model_full.fit(x_train, y_train)              # 원본 데이터로 학습

# PCA 있을 때 (축소된 특성으로)
model_pca = RandomForestClassifier(n_estimators=100, random_state=42)
model_pca.fit(x_train_pca, y_train)           # PCA로 줄인 데이터로 학습

#4. 평가 ──────────────────────────────────
acc_full = accuracy_score(y_test, model_full.predict(x_test))       # 원본 데이터 정확도
acc_pca = accuracy_score(y_test, model_pca.predict(x_test_pca))     # PCA 축소 후 정확도

print(f"\n[ 평가 — PCA 전후 비교 ]")
print(f"원본 (30D): accuracy={acc_full:.4f}")
print(f"PCA ({n_components}D):  accuracy={acc_pca:.4f}")
print(f"→ {30-n_components}개 특성을 줄여도 성능 {'유지' if abs(acc_full-acc_pca) < 0.02 else '변화'}")

# 누적 설명 비율
pca_full = PCA().fit(x_train)                 # 주성분 30개 전부 계산 (분석용)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)  # cumsum: 누적합 — 주성분 몇 개로 전체 분산의 몇 %를 설명하는지
n_95 = np.argmax(cum_var >= 0.95) + 1         # 95% 이상 설명되는 최소 주성분 수
print(f"\n[ 누적 설명 비율 ]")
for i in [2, 5, 10, n_95, 30]:
    print(f"  {i:2d}개: {cum_var[i-1]:.1%}")   # 주성분 개수별 누적 설명 비율

# 2D 시각화
pca_2d = PCA(n_components=2).fit_transform(x_train)  # 2개 주성분으로 축소 (2D 시각화용)

# ── 시각화 ────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))  # 1x3 격자 / figsize: 가로 15, 세로 4 인치

# (좌) Scree Plot — 주성분 개수별 누적 설명 비율
axes[0].plot(range(1, 31), cum_var, 'bo-', markersize=4)  # 주성분 1~30개의 누적 설명 비율
axes[0].axhline(0.95, color='red', linestyle='--', label='95%')    # axhline: 수평선 — 95% 기준선
axes[0].axvline(n_95, color='green', linestyle='--', label=f'n={n_95}')  # 95% 달성 주성분 수
axes[0].set_xlabel('Components')   # x축: 주성분 개수
axes[0].set_ylabel('Cumulative Variance')  # y축: 누적 설명 비율
axes[0].set_title('Scree Plot')    # Scree Plot: 주성분 개수 선택에 사용
axes[0].legend()                   # 범례

# (중) 2D 시각화 — 30차원 데이터를 2차원으로 축소해서 시각화
scatter = axes[1].scatter(pca_2d[:, 0], pca_2d[:, 1], c=y_train, cmap='coolwarm', s=20, alpha=0.7)  # c=y_train: 라벨별 색상 / alpha: 투명도
axes[1].set_xlabel('PC1')          # PC1: 첫 번째 주성분 (가장 많은 분산을 설명)
axes[1].set_ylabel('PC2')          # PC2: 두 번째 주성분
axes[1].set_title('Cancer 2D Visualization')  # 30차원이지만 2차원에서도 클래스가 구분됨

# (우) PCA 전후 정확도 비교
axes[2].bar(['Original\n(30D)', f'PCA\n({n_components}D)'], [acc_full, acc_pca],
            color=['gray', 'steelblue'], edgecolor='black')  # bar: 막대그래프 / color: 각 막대 색상
axes[2].set_ylim(0.9, 1.0)        # y축 범위 조정 (차이를 강조)
axes[2].set_title('Accuracy: Original vs PCA')

plt.tight_layout()                 # 여백 자동 조정
plt.savefig('3-7_output.png', dpi=100)  # 이미지 저장
plt.show()                         # 화면 표시

print("\n" + "="*50)
print("핵심 정리:")
print(f"  #2에 PCA 추가: 30D → {n_components}D (95% 설명)")
print("  스케일링 필수 (PCA는 분산 기반)")
print("  차원 줄여도 성능 유지 → 노이즈 제거 효과")
print("  2D 시각화로 데이터 구조 파악 가능")
print("="*50)
