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
import numpy as np                              # numpy: 숫자 배열(행렬) 연산 라이브러리
import matplotlib.pyplot as plt                  # matplotlib.pyplot: 그래프/차트 시각화 라이브러리
from sklearn.cluster import KMeans               # KMeans: K-평균 군집화 — 데이터를 K개 그룹으로 나누는 비지도학습 알고리즘
from sklearn.datasets import make_blobs, load_iris  # make_blobs: 뭉치(blob) 형태의 합성 데이터 생성 / load_iris: 붓꽃 데이터
from sklearn.preprocessing import StandardScaler    # StandardScaler: 정규화 (K-Means도 거리 기반이라 스케일링 권장)
from sklearn.metrics import adjusted_rand_score     # adjusted_rand_score: 군집 결과가 실제 라벨과 얼마나 일치하는지 (0=랜덤, 1=완벽)

#1. 데이터 ─────────────────────────────────
x, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)  # n_samples=300: 300개 데이터 / centers=4: 4개 군집 / cluster_std=0.8: 군집 내 퍼짐 정도

print(f"[ 합성 데이터 — 4개 군집 ]")
print(f"shape: {x.shape}")           # (300, 2) — 2차원 데이터

#2. 데이터 전처리 ──────────────────────────
# K-Means는 거리 기반이므로 스케일링 권장
# (make_blobs는 이미 비슷한 범위라 생략 가능)

#3. 모델 ──────────────────────────────────
model = KMeans(n_clusters=4, random_state=42, n_init=10)  # n_clusters=4: 4개 군집으로 나누기 / n_init=10: 초기 중심점을 10번 랜덤하게 시도해서 가장 좋은 것 선택
model.fit(x)                                               # fit: 데이터 점들을 반복적으로 가장 가까운 중심에 배정 → 중심 이동 → 수렴할 때까지

labels = model.labels_                # labels_: 각 데이터가 어떤 군집에 속하는지 (0, 1, 2, 3)
centers = model.cluster_centers_      # cluster_centers_: 각 군집의 중심 좌표

#4. 평가 ──────────────────────────────────
ari = adjusted_rand_score(y_true, labels)  # adjusted_rand_score: 군집 결과 vs 실제 라벨 비교 (1이면 완벽 일치)

print(f"\n[ 평가 ]")
print(f"군집 수: {len(set(labels))}")        # set: 중복 제거 → 몇 개 군집인지
print(f"Adjusted Rand Index: {ari:.4f} (1이면 완벽)")
print(f"Inertia: {model.inertia_:.1f}")     # inertia_: 각 데이터와 자기 군집 중심까지 거리의 합 (작을수록 좋음)

# Elbow Method
print(f"\n[ Elbow Method — K 선택 ]")        # Elbow Method: 최적 K(군집 수)를 찾는 방법 — Inertia가 급격히 꺾이는 지점이 최적
inertias = []                                # 각 K별 Inertia 저장
for k in range(1, 11):                       # K를 1~10까지 바꿔가며 실험
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(x)  # K개 군집으로 학습
    inertias.append(km.inertia_)             # Inertia 저장
    print(f"  K={k:2d}: Inertia={km.inertia_:8.1f}")  # K가 커지면 Inertia↓ (팔꿈치 지점에서 멈추면 됨)

# Iris에도 적용
iris = load_iris()                            # 붓꽃 데이터 로드
x_iris = StandardScaler().fit_transform(iris.data)  # 스케일링 적용
km_iris = KMeans(n_clusters=3, random_state=42, n_init=10).fit(x_iris)  # 3개 군집으로 학습 (붓꽃은 3품종이니까)
ari_iris = adjusted_rand_score(iris.target, km_iris.labels_)  # 실제 품종과 군집 결과 비교
print(f"\n[ Iris K-Means: ARI={ari_iris:.4f} ]")  # 정답 없이도 품종을 어느 정도 구분함

# ── 시각화 ────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))  # 1x3 격자 / figsize: 가로 15, 세로 4 인치

# (좌) 군집 결과 + 중심점
axes[0].scatter(x[:, 0], x[:, 1], c=labels, cmap='viridis', s=20)  # scatter: 산점도 / c=labels: 군집별 색상 / s: 점 크기
axes[0].scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, edgecolors='black')  # 군집 중심을 빨간 X로 표시 / s=200: 큰 점
axes[0].set_title(f'K-Means (K=4, ARI={ari:.3f})')

# (중) Elbow Method — K 선택 그래프
axes[1].plot(range(1, 11), inertias, 'bo-', markersize=6)  # 'bo-': 파란 동그라미+선
axes[1].axvline(4, color='red', linestyle='--', alpha=0.5)  # 최적 K=4 위치에 수직선
axes[1].set_xlabel('K')            # x축: 군집 수
axes[1].set_ylabel('Inertia')      # y축: 군집 내 거리 합 (작을수록 좋음)
axes[1].set_title('Elbow Method')  # 팔꿈치처럼 꺾이는 지점이 최적 K

# (우) Iris 군집 결과
axes[2].scatter(x_iris[:, 0], x_iris[:, 1], c=km_iris.labels_, cmap='viridis', s=20)  # Iris 군집 결과 시각화
axes[2].set_title(f'Iris K-Means (ARI={ari_iris:.3f})')

plt.tight_layout()                  # 여백 자동 조정
plt.savefig('3-6_output.png', dpi=100)  # 이미지 저장
plt.show()                          # 화면 표시

print("\n" + "="*50)
print("핵심 정리:")
print("  #3만 바뀜: KMeans(n_clusters=4)")
print("  비지도학습 = 정답(y) 없이 패턴 발견")
print("  Elbow Method = K를 선택하는 방법")
print("  ARI = 군집 결과와 실제 라벨 비교")
print("="*50)
