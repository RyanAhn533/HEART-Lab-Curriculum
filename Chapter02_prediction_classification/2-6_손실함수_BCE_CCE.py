# ============================================
# 2-6. 분류 손실함수 — BCE / CCE
#
# 왜 배우는가:
#   분류에서는 MSE 대신 Cross-Entropy를 쓴다.
#   문제 유형 보고 loss를 선택하는 기준.
#
# 나중에 만나는 곳:
#   → Chapter 4: loss='binary_crossentropy' / 'categorical_crossentropy'
#
# ▶ 보고 오기: StatQuest "Cross Entropy"
#
# Ref: Coursera C1W3 / Google MLCC
# ============================================

import numpy as np                                    # numpy: 수학 계산 라이브러리 (로그, 지수, 배열 등)
import matplotlib.pyplot as plt                        # matplotlib: 그래프를 그리는 라이브러리

# ── 1. BCE 직접 계산 ─────────────────────
print("[ Binary Cross-Entropy 직접 계산 ]")

def bce(y_true, y_pred):                               # BCE 함수: 이진분류의 손실을 계산
    epsilon = 1e-15                                    # epsilon: 아주 작은 수 (log(0)이 되면 무한대가 되니까 방지)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)     # clip: 예측값을 epsilon~(1-epsilon) 범위로 제한
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))  # BCE 공식: 정답이 1이면 -log(p), 0이면 -log(1-p)의 평균

y_true = np.array([1, 1, 0, 0, 1])                    # 실제 라벨: 1=양성, 0=음성

# 좋은 예측
y_good = np.array([0.95, 0.90, 0.05, 0.10, 0.85])    # 좋은 예측: 실제 1에 높은 확률, 실제 0에 낮은 확률
# 나쁜 예측
y_bad = np.array([0.30, 0.40, 0.80, 0.70, 0.20])     # 나쁜 예측: 실제와 반대 방향의 확률

bce_good = bce(y_true, y_good)                         # 좋은 예측의 BCE (낮아야 좋음)
bce_bad = bce(y_true, y_bad)                           # 나쁜 예측의 BCE (높을수록 나쁨)

print(f"실제:     {y_true}")
print(f"좋은 예측: {y_good} → BCE = {bce_good:.4f}")
print(f"나쁜 예측: {y_bad}  → BCE = {bce_bad:.4f}")
print(f"→ 잘 맞출수록 BCE가 낮다")

# ── 2. 확신하며 틀리면 페널티 ────────────
print(f"\n[ 확신하며 틀리면 페널티가 극대화 ]")
probs = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]        # 다양한 예측 확률
for p in probs:                                        # 각 확률에 대해
    loss_when_true = -np.log(p)                        # 실제=1일 때 손실: 확률이 낮을수록 손실 큼 (자신있게 틀리면 큰 벌)
    loss_when_false = -np.log(1 - p)                   # 실제=0일 때 손실: 확률이 높을수록 손실 큼
    print(f"  p={p:.2f} → 실제1이면 loss={loss_when_true:.3f}, 실제0이면 loss={loss_when_false:.3f}")

# ── 3. CCE 직접 계산 ─────────────────────
print(f"\n[ Categorical Cross-Entropy 직접 계산 ]")

def cce(y_true_onehot, y_pred_softmax):                # CCE 함수: 다중분류의 손실을 계산 (샘플 1개 기준)
    epsilon = 1e-15                                    # log(0) 방지용 작은 수
    y_pred_softmax = np.clip(y_pred_softmax, epsilon, 1.0)  # 예측값을 epsilon 이상으로 제한
    return -np.sum(y_true_onehot * np.log(y_pred_softmax))  # 표준 정의: 샘플별 CCE = -Σ y·log(p)
    # ↑ 클래스 수로 나누면 안 됨! 배치(샘플 여러 개)일 때만 "샘플 수"로 평균을 낸다

# 3클래스 (정답: 클래스 0)
y_true_oh = np.array([1, 0, 0])                        # One-Hot 인코딩: 클래스 0이 정답 ([1,0,0])
y_good_sm = np.array([0.85, 0.10, 0.05])              # 좋은 예측: 클래스 0에 높은 확률
y_bad_sm = np.array([0.20, 0.50, 0.30])               # 나쁜 예측: 클래스 0에 낮은 확률
# 검산: One-Hot이라 정답 클래스 항만 남는다 → CCE = -log(0.85) ≈ 0.163 이 나와야 정상

print(f"One-Hot 실제:  {y_true_oh}")
print(f"좋은 Softmax:  {y_good_sm} → CCE = {cce(y_true_oh, y_good_sm):.4f}")
print(f"나쁜 Softmax:  {y_bad_sm}  → CCE = {cce(y_true_oh, y_bad_sm):.4f}")

# ── 4. MSE vs BCE 비교 ──────────────────
print(f"\n[ MSE vs BCE — 왜 분류에 MSE 안 쓰는가 ]")
mse_good = np.mean((y_true - y_good)**2)              # 좋은 예측의 MSE
mse_bad = np.mean((y_true - y_bad)**2)                # 나쁜 예측의 MSE
print(f"MSE 좋은 예측: {mse_good:.4f}, MSE 나쁜 예측: {mse_bad:.4f}")
print(f"BCE 좋은 예측: {bce_good:.4f}, BCE 나쁜 예측: {bce_bad:.4f}")
print(f"→ BCE가 좋은/나쁜 예측의 차이를 더 크게 만든다")  # BCE가 분류에 더 적합한 이유

# ── 5. 시각화 ─────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))        # 2행 3열 그래프 영역

# BCE: p vs loss (실제=1일 때)
p_range = np.linspace(0.01, 0.99, 100)                 # 0.01~0.99 범위의 예측 확률
loss_true1 = -np.log(p_range)                          # 실제=1일 때 손실: 확률 높을수록 손실 작음
loss_true0 = -np.log(1 - p_range)                      # 실제=0일 때 손실: 확률 낮을수록 손실 작음
axes[0, 0].plot(p_range, loss_true1, 'b-', linewidth=2, label='Actual=1')  # 실제=1일 때 손실 곡선
axes[0, 0].plot(p_range, loss_true0, 'r-', linewidth=2, label='Actual=0')  # 실제=0일 때 손실 곡선
axes[0, 0].set_title('BCE Loss vs Predicted Probability')  # 제목
axes[0, 0].set_xlabel('Predicted Probability')         # x축: 예측 확률
axes[0, 0].set_ylabel('Loss')                          # y축: 손실값
axes[0, 0].legend()                                    # 범례

# 좋은 예측 vs 나쁜 예측
axes[0, 1].bar(['Good Prediction', 'Bad Prediction'], [bce_good, bce_bad],
               color=['green', 'red'], edgecolor='black')  # 좋은/나쁜 예측의 BCE 비교 막대 그래프
axes[0, 1].set_title('BCE: Good vs Bad')               # 제목
axes[0, 1].set_ylabel('BCE Loss')                      # y축: BCE 값

# MSE vs BCE 비교
x_comp = ['MSE Good', 'MSE Bad', 'BCE Good', 'BCE Bad']  # 비교 항목 이름
y_comp = [mse_good, mse_bad, bce_good, bce_bad]       # 각 값
colors = ['lightgreen', 'lightsalmon', 'green', 'red'] # 색상
axes[0, 2].bar(x_comp, y_comp, color=colors, edgecolor='black')  # MSE와 BCE 나란히 비교
axes[0, 2].set_title('MSE vs BCE Comparison')          # 제목
axes[0, 2].tick_params(axis='x', rotation=30)          # x축 라벨 30도 회전 (겹침 방지)

# 손실 곡면: MSE로 sigmoid 최적화 (울퉁불퉁)
w_range = np.linspace(-5, 5, 100)                      # w 후보 범위
x_data = np.array([1, 2, 3])                           # 간단한 입력 데이터
y_data = np.array([0, 0, 1])                           # 정답 라벨

def sigmoid(z):                                        # sigmoid 함수 (재정의)
    return 1 / (1 + np.exp(-z))                        # 입력을 0~1로 변환

mse_surface = [np.mean((y_data - sigmoid(w * x_data))**2) for w in w_range]  # 각 w에서 MSE 계산 (울퉁불퉁한 곡면)
bce_surface = []                                       # BCE 곡면 저장용 리스트
for w in w_range:                                      # 각 w에 대해
    p = np.clip(sigmoid(w * x_data), 1e-15, 1 - 1e-15)  # sigmoid 출력 (log(0) 방지)
    bce_val = -np.mean(y_data * np.log(p) + (1 - y_data) * np.log(1 - p))  # BCE 계산
    bce_surface.append(bce_val)                        # 결과 저장

axes[1, 0].plot(w_range, mse_surface, 'r-', linewidth=2)  # MSE 손실 곡면 (빨간색): 비볼록 + 양끝이 평평
axes[1, 0].set_title('MSE Loss Surface (non-convex, flat tails)')  # 제목: sigmoid 포화 구간에서 기울기≈0 → 학습 매우 느림
axes[1, 0].set_xlabel('w')                             # x축: 가중치
axes[1, 0].set_ylabel('Loss')                          # y축: 손실

axes[1, 1].plot(w_range, bce_surface, 'b-', linewidth=2)  # BCE 손실 곡면 (파란색, 볼록)
axes[1, 1].set_title('BCE Loss Surface (convex)')      # 제목: BCE는 볼록 + 틀릴수록 기울기 큼 → 경사하강법이 잘 작동
axes[1, 1].set_xlabel('w')                             # x축: 가중치
axes[1, 1].set_ylabel('Loss')                          # y축: 손실

# 문제 유형별 정리
table_data = [                                         # 문제 유형 → 활성화 → 손실함수 정리표
    ['Regression', 'None', 'MSE'],                     # 회귀: 활성화 없음, MSE 사용
    ['Binary Clf', 'Sigmoid', 'BCE'],                  # 이진분류: Sigmoid + BCE
    ['Multi Clf', 'Softmax', 'CCE'],                   # 다중분류: Softmax + CCE
]
axes[1, 2].axis('off')                                 # 축 숨기기 (표를 그릴 영역)
table = axes[1, 2].table(cellText=table_data,          # table: 표 그리기
                          colLabels=['Problem', 'Activation', 'Loss'],  # 열 제목
                          loc='center', cellLoc='center')  # 가운데 정렬
table.auto_set_font_size(False)                        # 자동 글자 크기 비활성화
table.set_fontsize(12)                                 # 글자 크기 12
table.scale(1, 2)                                      # 표 세로 크기 2배
axes[1, 2].set_title('Problem → Loss Mapping', pad=20)  # 제목 (pad: 제목과 표 사이 간격)

plt.tight_layout()                                     # 그래프 간 간격 자동 조정
plt.savefig('2-6_output.png', dpi=100)                # 이미지 파일로 저장
plt.show()                                             # 화면에 표시

# ── 6. 고리 닫기 — GD(2-4) + BCE(2-6) = 로지스틱 회귀(2-5) ──
print(f"\n[ 고리 닫기: numpy 로지스틱 회귀 직접 구현 ]")
np.random.seed(42)                                     # 난수 시드 고정
X1 = np.concatenate([np.random.normal(-1, 1, 50),      # 클래스 0: 평균 -1 주변 50개
                     np.random.normal(1, 1, 50)])       # 클래스 1: 평균 +1 주변 50개
y1 = np.array([0]*50 + [1]*50)                          # 정답 라벨

w1, b1, lr = 0.0, 0.0, 0.5                              # 초기 가중치/절편/학습률
for epoch in range(2000):                               # GD 반복
    p1 = sigmoid(w1 * X1 + b1)                          # 순전파: 선형회귀 + sigmoid = 확률
    dw = np.mean((p1 - y1) * X1)                        # BCE를 w로 미분하면 (p-y)·x — 깔끔한 형태!
    db = np.mean(p1 - y1)                               # BCE를 b로 미분하면 (p-y)
    w1 -= lr * dw                                       # 2-4에서 배운 업데이트 공식 그대로
    b1 -= lr * db

from sklearn.linear_model import LogisticRegression     # 비교용 sklearn 모델
sk = LogisticRegression(C=1e5).fit(X1.reshape(-1, 1), y1)  # C=1e5: 정규화를 거의 꺼서 순수 BCE 최소화와 비교
acc_np = np.mean((sigmoid(w1 * X1 + b1) >= 0.5) == y1)  # numpy 구현의 정확도
print(f"  numpy GD:  w={w1:.3f}, b={b1:.3f}, acc={acc_np:.3f}")
print(f"  sklearn:   w={sk.coef_[0][0]:.3f}, b={sk.intercept_[0]:.3f}, acc={sk.score(X1.reshape(-1, 1), y1):.3f}")
print(f"  → GD로 BCE를 최소화하면 로지스틱 회귀(2-5)가 된다")

# ── 정리 ──────────────────────────────────
print("\n" + "="*50)
print("핵심 정리:")
print("  회귀 → MSE (loss='mse')")
print("  이진분류 → BCE (loss='binary_crossentropy')")
print("  다중분류 → CCE (loss='categorical_crossentropy')")
print("  BCE는 확신하며 틀리면 페널티 극대화")
print("  분류에 MSE 안 쓰는 이유: 비볼록 + 포화 구간 기울기 소실")
print("  GD(2-4) + BCE(2-6) = 로지스틱 회귀(2-5)")
print("="*50)
