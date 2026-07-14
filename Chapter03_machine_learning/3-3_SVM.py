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
import numpy as np                              # numpy: 숫자 배열(행렬) 연산 라이브러리
import matplotlib.pyplot as plt                  # matplotlib.pyplot: 그래프/차트 시각화 라이브러리
from sklearn.svm import SVC                      # SVC: Support Vector Classifier — 데이터를 가장 넓은 마진(여유)으로 나누는 분류 모델
from sklearn.datasets import load_breast_cancer, make_moons  # load_breast_cancer: 유방암 데이터 / make_moons: 반달 모양 비선형 합성 데이터
from sklearn.model_selection import train_test_split          # train_test_split: 데이터를 훈련/테스트로 분할
from sklearn.preprocessing import StandardScaler              # StandardScaler: 평균=0, 표준편차=1로 정규화 — SVM은 거리 기반이라 스케일링 필수!
from sklearn.metrics import accuracy_score, classification_report  # accuracy_score: 정확도 / classification_report: 상세 평가

#1. 데이터 ─────────────────────────────────
data = load_breast_cancer()   # 유방암 데이터 로드 (569명, 30개 측정값)
x = data.data                 # x: 입력 특성 (569, 30)
y = data.target               # y: 정답 (0=악성, 1=양성)

print(f"[ Breast Cancer — SVM ]")
print(f"shape: {x.shape}")

#2. 데이터 전처리 및 분할 ───────────────────
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42    # 20% 테스트 / random_state: 재현성
)
scaler = StandardScaler()                    # StandardScaler 객체 생성 — 각 특성의 평균을 0, 표준편차를 1로 맞춤
x_train = scaler.fit_transform(x_train)      # fit_transform: 훈련 데이터로 평균/표준편차 계산(fit) + 변환(transform) 동시에
x_test = scaler.transform(x_test)            # transform: 훈련 때 계산한 평균/표준편차로 테스트 데이터도 같은 기준으로 변환 (fit하면 안 됨!)

#3. 모델 ──────────────────────────────────
model = SVC(kernel='rbf', C=1.0)             # kernel='rbf': 방사형 기저함수 커널 (비선형 경계를 만들 수 있음, 가장 많이 사용) / C=1.0: 규제 강도의 역수 — C 클수록 마진 위반에 엄격(규제 약함 → 과적합 위험), 작을수록 위반에 관대(규제 강함 → 과소적합 위험)
model.fit(x_train, y_train)                   # fit: 서포트 벡터(경계에 가장 가까운 데이터)를 찾아 최적 분류 경계를 학습

#4. 평가 ──────────────────────────────────
y_pred = model.predict(x_test)               # predict: 학습된 경계를 기준으로 새 데이터 분류
acc = accuracy_score(y_test, y_pred)          # 정확도 계산

print(f"\n[ 평가 ]")
print(f"Accuracy: {acc:.4f}")
print(f"서포트 벡터 수: {model.n_support_}")    # n_support_: 각 클래스별 서포트 벡터(경계 근처 핵심 데이터) 개수
print(f"\n{classification_report(y_test, y_pred, target_names=['Malignant', 'Benign'])}")

# ── 추가: 커널 비교 (비선형 데이터) ──────
print("[ 커널 비교 — Moon 데이터 (비선형) ]")
x_moon, y_moon = make_moons(n_samples=300, noise=0.2, random_state=42)  # make_moons: 반달 모양 데이터 300개 생성 / noise=0.2: 데이터에 약간의 랜덤 노이즈 추가
xm_tr, xm_te, ym_tr, ym_te = train_test_split(x_moon, y_moon, test_size=0.2, random_state=42)

for k in ['linear', 'rbf', 'poly']:          # 3가지 커널 비교: linear(직선), rbf(곡선), poly(다항식)
    svm = SVC(kernel=k, C=1.0).fit(xm_tr, ym_tr)  # 각 커널로 SVM 학습
    acc_k = accuracy_score(ym_te, svm.predict(xm_te))
    print(f"  kernel={k:8s}: {acc_k:.4f}")    # 비선형 데이터에서는 rbf가 linear보다 좋음

# ── 시각화 ────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))  # 1x3 격자 / figsize: 가로 15, 세로 4 인치

def plot_boundary(ax, model, X, y, title):
    """결정 경계를 시각화하는 함수"""
    h = 0.02                                  # h: 격자 간격 (작을수록 경계가 부드러움)
    xx, yy = np.meshgrid(np.arange(X[:,0].min()-0.5, X[:,0].max()+0.5, h),  # meshgrid: 2D 격자 좌표 생성
                          np.arange(X[:,1].min()-0.5, X[:,1].max()+0.5, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)  # 격자의 모든 점을 예측 → 결정 경계 생성
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')  # contourf: 결정 영역 색칠 / cmap='coolwarm': 빨강-파랑 색상맵
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=20, edgecolors='k')  # scatter: 데이터 점 / s: 점 크기 / edgecolors: 테두리
    ax.set_title(title)

for i, k in enumerate(['linear', 'rbf', 'poly']):  # enumerate: 인덱스(i)와 커널이름(k)을 동시에 가져옴
    svm = SVC(kernel=k, C=1.0).fit(xm_tr, ym_tr)   # 각 커널로 학습
    acc_k = accuracy_score(ym_te, svm.predict(xm_te))
    plot_boundary(axes[i], svm, xm_tr, ym_tr, f'{k} (acc={acc_k:.3f})')  # 각 커널의 결정 경계 시각화

plt.tight_layout()                           # 여백 자동 조정
plt.savefig('3-3_output.png', dpi=100)       # 이미지 저장
plt.show()                                   # 화면 표시

# ── 정리 ──────────────────────────────────
print("\n" + "="*50)
print("핵심 정리:")
print("  #3만 바뀜: SVC(kernel='rbf', C=1.0)")
print("  #2에 StandardScaler 추가 (SVM은 스케일링 필수!)")
print("  kernel='rbf' → 대부분 잘 작동")
print("  C 크면 과적합, 작으면 과소적합")
print("="*50)
