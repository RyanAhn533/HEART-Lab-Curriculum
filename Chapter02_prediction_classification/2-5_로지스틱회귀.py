# ============================================
# 2-5. 로지스틱 회귀 (Logistic Regression)
#
# 왜 배우는가:
#   분류 문제의 시작. 선형회귀 + Sigmoid = 분류 모델.
#   모든 신경망 이진분류의 기초.
#
# 나중에 만나는 곳:
#   → Chapter 4: activation='sigmoid' / 'softmax'
#   → 2-6: BCE (이진분류 손실함수)
#
# ▶ 보고 오기: Coursera C1W3 "Logistic Regression"
#
# Ref: Stanford CS229 W2 / Google MLCC
# ============================================

import numpy as np                                    # numpy: 수학 계산 라이브러리 (배열, 지수함수 등)
import matplotlib.pyplot as plt                        # matplotlib: 그래프를 그리는 라이브러리
from sklearn.linear_model import LogisticRegression    # LogisticRegression: 분류(0/1)를 위한 모델 (이름에 회귀가 들어가지만 분류 모델!)
from sklearn.datasets import load_breast_cancer, load_iris  # 유방암/붓꽃 예제 데이터셋
from sklearn.model_selection import train_test_split   # train_test_split: 데이터를 학습/테스트로 분할
from sklearn.preprocessing import StandardScaler       # StandardScaler: 특성값을 평균0, 표준편차1로 정규화
from sklearn.metrics import accuracy_score             # accuracy_score: 전체 중 맞춘 비율 계산

# ── 1. Sigmoid 함수 이해 ─────────────────
print("[ Sigmoid 함수 ]")

def sigmoid(z):                                        # sigmoid: 어떤 숫자든 0~1 사이 값으로 변환하는 함수
    return 1 / (1 + np.exp(-z))                        # 수식: 1 / (1 + e^(-z))

z_values = np.linspace(-10, 10, 100)                   # -10~10 사이 100개 점 생성
sig_values = sigmoid(z_values)                         # 각 z에 대해 sigmoid 계산

print(f"z=-10 → sigmoid={sigmoid(-10):.6f} (거의 0)")  # 큰 음수 → 0에 가까움
print(f"z=  0 → sigmoid={sigmoid(0):.1f}    (정확히 0.5)")  # 0 → 정확히 0.5
print(f"z=+10 → sigmoid={sigmoid(10):.6f} (거의 1)")  # 큰 양수 → 1에 가까움

# ── 2. 이진분류 (Breast Cancer) ──────────
print(f"\n[ 이진분류 — Breast Cancer ]")
data = load_breast_cancer()                            # 유방암 데이터셋 로드 (30개 특성, 569개 샘플)
X = data.data                                          # 특성 데이터 (세포 크기, 모양 등 30개)
y = data.target                                        # 정답 라벨: 0=악성, 1=양성

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 80% 학습, 20% 테스트
scaler = StandardScaler()                              # 정규화 객체 생성 (특성별로 스케일을 맞춤)
X_train = scaler.fit_transform(X_train)                # fit_transform: 학습 데이터에서 평균/표준편차 계산 후 정규화 적용
X_test = scaler.transform(X_test)                      # transform: 학습 때 계산한 평균/표준편차로 테스트 데이터 정규화 (fit 없이!)

model = LogisticRegression(max_iter=1000)              # 로지스틱 회귀 모델 생성 (max_iter: 최적화 반복 최대 횟수)
model.fit(X_train, y_train)                            # fit: 데이터로 학습 (각 특성의 가중치를 찾아 분류 경계 결정)
y_pred = model.predict(X_test)                         # predict: 테스트 데이터의 클래스(0 또는 1) 예측
y_prob = model.predict_proba(X_test)[:, 1]             # predict_proba: 각 클래스일 확률 반환, [:, 1]은 양성(1)일 확률만 추출

acc = accuracy_score(y_test, y_pred)                   # accuracy_score: 정확도 = 전체 중 맞춘 비율. 자세한 평가지표는 2-7에서
print(f"Accuracy: {acc:.4f}")
print(f"예측 확률 (처음 5개): {y_prob[:5].round(3)}")  # 처음 5개 샘플의 양성 확률
print(f"예측 클래스: {y_pred[:5]}")                    # 처음 5개의 예측 라벨
print(f"실제 클래스: {y_test[:5]}")                    # 처음 5개의 실제 라벨

# ── 3. 다중분류 (Iris) ───────────────────
print(f"\n[ 다중분류 — Iris (3 클래스) ]")
iris = load_iris()                                     # 붓꽃 데이터셋 로드 (4개 특성, 3종류, 150개 샘플)
X_iris = iris.data[:, :2]                              # 시각화를 위해 처음 2개 특성만 사용 (꽃받침 길이, 너비)
y_iris = iris.target                                   # 정답: 0, 1, 2 (3종류 붓꽃)

X_tr, X_te, y_tr, y_te = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)  # 학습/테스트 분할
scaler2 = StandardScaler()                             # 두 번째 정규화 객체
X_tr = scaler2.fit_transform(X_tr)                     # 학습 데이터 정규화
X_te = scaler2.transform(X_te)                         # 테스트 데이터 정규화

model_multi = LogisticRegression(max_iter=1000)        # 다중분류용 로지스틱 회귀 (자동으로 다중분류 처리)
model_multi.fit(X_tr, y_tr)                            # fit: 3개 클래스를 구분하는 경계 학습
acc_multi = model_multi.score(X_te, y_te)              # score: 정확도를 바로 계산해서 반환 (predict + accuracy_score 한번에)
print(f"Accuracy: {acc_multi:.4f}")

# ── 4. 시각화 ─────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))        # 2행 3열 그래프 영역

# Sigmoid 함수
axes[0, 0].plot(z_values, sig_values, 'b-', linewidth=2)  # sigmoid 곡선 (S자 모양)
axes[0, 0].axhline(0.5, color='red', linestyle='--', alpha=0.5, label='threshold=0.5')  # 0.5 기준선: 이 위면 양성, 아래면 음성
axes[0, 0].axvline(0, color='gray', linestyle=':', alpha=0.5)  # z=0 기준선
axes[0, 0].fill_between(z_values, sig_values, 0.5, where=(sig_values >= 0.5),
                         alpha=0.2, color='green', label='Positive')   # 0.5 이상 영역 초록색 = 양성
axes[0, 0].fill_between(z_values, sig_values, 0.5, where=(sig_values < 0.5),
                         alpha=0.2, color='red', label='Negative')     # 0.5 미만 영역 빨간색 = 음성
axes[0, 0].set_title('Sigmoid Function')               # 제목
axes[0, 0].set_xlabel('z')                             # x축: 입력값
axes[0, 0].set_ylabel('σ(z)')                          # y축: sigmoid 출력 (확률)
axes[0, 0].legend(fontsize=8)                          # 범례 (글자 크기 8)

# 예측 확률 분포
axes[0, 1].hist(y_prob[y_test == 1], bins=15, alpha=0.7, label='Positive', color='green')  # 실제 양성의 예측 확률 분포 (1에 가까워야 좋음)
axes[0, 1].hist(y_prob[y_test == 0], bins=15, alpha=0.7, label='Negative', color='red')    # 실제 음성의 예측 확률 분포 (0에 가까워야 좋음)
axes[0, 1].axvline(0.5, color='black', linestyle='--', label='Threshold')  # 분류 기준선 0.5
axes[0, 1].set_title('Predicted Probabilities')        # 제목: 예측 확률
axes[0, 1].set_xlabel('Probability')                   # x축: 확률
axes[0, 1].legend()                                    # 범례

# 실제 vs 예측
axes[0, 2].scatter(range(len(y_test)), y_test, color='blue', s=30, label='Actual', alpha=0.7)    # 실제 라벨
axes[0, 2].scatter(range(len(y_pred)), y_pred, color='red', s=10, label='Predicted', alpha=0.7)  # 예측 라벨
axes[0, 2].set_title(f'Actual vs Predicted (Acc={acc:.3f})')  # 제목에 정확도 표시
axes[0, 2].set_ylabel('Class')                         # y축: 클래스 (0 또는 1)
axes[0, 2].legend()                                    # 범례

# 결정경계 (Iris 2D)
h = 0.02                                              # 격자 간격 (작을수록 세밀)
x_min, x_max = X_tr[:, 0].min() - 1, X_tr[:, 0].max() + 1  # x축 범위 (여백 1 추가)
y_min, y_max = X_tr[:, 1].min() - 1, X_tr[:, 1].max() + 1  # y축 범위 (여백 1 추가)
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))  # 2D 격자 생성 (결정경계 그리기용)
Z = model_multi.predict(np.c_[xx.ravel(), yy.ravel()])  # 격자의 모든 점에 대해 클래스 예측
Z = Z.reshape(xx.shape)                                # 예측 결과를 격자 모양으로 복원
axes[1, 0].contourf(xx, yy, Z, alpha=0.3, cmap='viridis')  # contourf: 결정경계를 색으로 표현
scatter = axes[1, 0].scatter(X_tr[:, 0], X_tr[:, 1], c=y_tr, cmap='viridis', s=30, edgecolors='k')  # 학습 데이터 점 (색=클래스)
axes[1, 0].set_title('Decision Boundary (Iris)')       # 제목: 결정경계
axes[1, 0].set_xlabel('Feature 1')                     # x축: 특성 1
axes[1, 0].set_ylabel('Feature 2')                     # y축: 특성 2

# Softmax 시각화
def softmax(z):                                        # softmax: 여러 값을 확률 분포로 변환 (합=1)
    exp_z = np.exp(z - np.max(z))                      # overflow 방지를 위해 최댓값을 빼고 지수 계산
    return exp_z / exp_z.sum()                         # 각 지수값을 전체 합으로 나눔 → 확률

logits = np.array([2.0, 1.0, 0.1])                    # 모델의 원시 출력값 (logit) 3개
probs = softmax(logits)                                # softmax로 확률 변환 (합=1)
axes[1, 1].bar(['Class 0', 'Class 1', 'Class 2'], probs,
               color=['steelblue', 'green', 'orange'], edgecolor='black')  # 각 클래스 확률 막대 그래프
axes[1, 1].set_title(f'Softmax Output\n{probs.round(3)}')  # 제목에 확률값 표시
axes[1, 1].set_ylabel('Probability')                   # y축: 확률

# 선형회귀 vs 로지스틱 비교
x_demo = np.linspace(-3, 3, 100)                       # -3~3 범위의 입력값
axes[1, 2].plot(x_demo, x_demo, 'b--', label='Linear (unbounded)', alpha=0.7)  # 선형: 출력이 무한대까지 갈 수 있음
axes[1, 2].plot(x_demo, sigmoid(x_demo), 'r-', linewidth=2, label='Logistic (0~1)')  # 로지스틱: 출력이 항상 0~1
axes[1, 2].set_title('Linear vs Logistic')             # 제목: 선형 vs 로지스틱 비교
axes[1, 2].legend()                                    # 범례
axes[1, 2].set_xlabel('Input')                         # x축: 입력
axes[1, 2].set_ylabel('Output')                        # y축: 출력

plt.tight_layout()                                     # 그래프 간 간격 자동 조정
plt.savefig('2-5_output.png', dpi=100)               # 이미지 파일로 저장
plt.show()                                             # 화면에 표시

# ── 정리 ──────────────────────────────────
print("\n" + "="*50)
print("핵심 정리:")
print("  로지스틱 회귀 = 선형회귀 + Sigmoid")
print("  Sigmoid → 출력을 0~1 확률로 압축")
print("  threshold 0.5 → 이진분류")
print("  Softmax → 다중분류 (확률 합 = 1)")
print("  → 이제 이 분류를 '평가'하는 방법이 필요 (2-7)")
print("="*50)
