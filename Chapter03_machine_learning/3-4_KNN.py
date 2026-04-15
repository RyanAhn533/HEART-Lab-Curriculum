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
import numpy as np                              # numpy: 숫자 배열(행렬) 연산 라이브러리
import matplotlib.pyplot as plt                  # matplotlib.pyplot: 그래프/차트 시각화 라이브러리
from sklearn.neighbors import KNeighborsClassifier  # KNeighborsClassifier: K-최근접이웃 분류기 — 새 데이터와 가장 가까운 K개 이웃의 다수결로 분류
from sklearn.datasets import load_iris           # load_iris: 붓꽃 예제 데이터 (4개 특성 → 3품종 분류)
from sklearn.model_selection import train_test_split  # train_test_split: 데이터를 훈련/테스트로 분할
from sklearn.preprocessing import StandardScaler      # StandardScaler: 평균=0, 표준편차=1 정규화 — KNN은 거리 기반이라 스케일링 필수!
from sklearn.metrics import accuracy_score, classification_report  # accuracy_score: 정확도 / classification_report: 상세 평가

#1. 데이터 ─────────────────────────────────
data = load_iris()            # Iris 데이터 로드 (150송이 붓꽃)
x = data.data                 # x: 입력 특성 (150, 4) — 꽃받침/꽃잎 길이와 너비
y = data.target               # y: 정답 라벨 (0, 1, 2)

print(f"[ Iris — KNN ]")
print(f"shape: {x.shape}")

#2. 데이터 전처리 및 분할 ───────────────────
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42    # 20% 테스트 / random_state: 재현성
)
scaler = StandardScaler()                    # KNN은 거리 기반이라 스케일링 필수 (예: cm와 mm가 섞이면 거리 계산이 왜곡됨)
x_train = scaler.fit_transform(x_train)      # fit_transform: 훈련 데이터 기준으로 평균/표준편차 계산 + 변환
x_test = scaler.transform(x_test)            # transform: 같은 기준으로 테스트 데이터 변환

#3. 모델 ──────────────────────────────────
model = KNeighborsClassifier(n_neighbors=5)   # n_neighbors=5: 새 데이터 주변 5개 이웃을 보고 다수결로 분류 (K값)
model.fit(x_train, y_train)                    # fit: KNN은 실제로 학습하지 않고, 훈련 데이터를 그대로 저장만 함 (게으른 학습)

#4. 평가 ──────────────────────────────────
y_pred = model.predict(x_test)               # predict: 테스트 데이터 각각에 대해, 훈련 데이터 중 가장 가까운 5개를 찾아 다수결
acc = accuracy_score(y_test, y_pred)          # 정확도 계산

print(f"\n[ 평가 ]")
print(f"Accuracy: {acc:.4f} (K=5)")
print(f"\n{classification_report(y_test, y_pred, target_names=data.target_names)}")  # 품종별 상세 평가

# K값별 성능
print("[ K값별 성능 ]")
k_range = range(1, 21)                       # K를 1부터 20까지 바꿔가며 실험
k_accs = []                                   # 각 K별 정확도 저장
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k).fit(x_train, y_train)  # K값을 바꿔가며 학습
    k_accs.append(accuracy_score(y_test, knn.predict(x_test)))       # 각 K의 정확도 저장
    if k <= 10:
        print(f"  K={k:2d}: {k_accs[-1]:.4f}")  # K=1~10 결과 출력

best_k = list(k_range)[np.argmax(k_accs)]    # np.argmax: 정확도가 가장 높은 K값 찾기
print(f"\n최적 K: {best_k} (acc={max(k_accs):.4f})")

# ── 시각화 ────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))  # 1x3 격자로 3개 그래프 / figsize: 가로 15, 세로 4 인치

# (좌) K값에 따른 정확도 변화
axes[0].plot(list(k_range), k_accs, 'b-o', markersize=5)  # 'b-o': 파란색 선+동그라미 / markersize: 점 크기
axes[0].axvline(best_k, color='red', linestyle='--', label=f'Best K={best_k}')  # axvline: 최적 K에 수직선 표시
axes[0].set_xlabel('K')           # x축: K값
axes[0].set_ylabel('Accuracy')    # y축: 정확도
axes[0].set_title('K vs Accuracy')  # K가 너무 작으면 과적합, 너무 크면 과소적합
axes[0].legend()                   # 범례 표시

def plot_knn(ax, k, X, y, title):
    """KNN 결정 경계를 시각화하는 함수"""
    knn = KNeighborsClassifier(n_neighbors=k).fit(X, y)  # K개 이웃으로 학습
    h = 0.02                                  # 격자 간격
    xx, yy = np.meshgrid(np.arange(X[:,0].min()-0.5, X[:,0].max()+0.5, h),  # 2D 격자 생성
                          np.arange(X[:,1].min()-0.5, X[:,1].max()+0.5, h))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)  # 격자의 모든 점을 예측
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')   # contourf: 결정 영역 색칠
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=20, edgecolors='k')  # 데이터 점 표시
    ax.set_title(title)

x_2d = x_train[:, :2]            # 2D 시각화를 위해 처음 2개 특성만 사용
plot_knn(axes[1], 1, x_2d, y_train, 'K=1 (Overfitting)')    # K=1: 과적합 — 경계가 울퉁불퉁 (가장 가까운 1개만 보니 노이즈에 민감)
plot_knn(axes[2], best_k, x_2d, y_train, f'K={best_k} (Good)')  # 최적 K: 경계가 부드러움 (여러 이웃 평균이라 안정적)

plt.tight_layout()                # 여백 자동 조정
plt.savefig('3-4_output.png', dpi=100)  # 이미지 저장
plt.show()                        # 화면 표시

# ── 정리 ──────────────────────────────────
print("\n" + "="*50)
print("핵심 정리:")
print(f"  #3만 바뀜: KNeighborsClassifier(n_neighbors={best_k})")
print("  #2에 StandardScaler (거리 기반이라 필수)")
print("  K=1 과적합, K 크면 과소적합")
print("  학습 없음 → 데이터 많으면 느림")
print("="*50)
