# ============================================
# 1-11. 분류 손실함수 — BCE / CCE
#
# 왜 배우는가:
#   분류에서는 MSE 대신 Cross-Entropy를 쓴다.
#   문제 유형 보고 loss를 선택하는 기준.
#
# 나중에 만나는 곳:
#   → Phase 5~6: loss='binary_crossentropy' / 'categorical_crossentropy'
#
# ▶ 보고 오기: StatQuest "Cross Entropy"
#
# Ref: Coursera C1W3 / Google MLCC
# ============================================

import numpy as np
import matplotlib.pyplot as plt

# ── 1. BCE 직접 계산 ─────────────────────
print("[ Binary Cross-Entropy 직접 계산 ]")

def bce(y_true, y_pred):
    epsilon = 1e-15  # log(0) 방지
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

y_true = np.array([1, 1, 0, 0, 1])

# 좋은 예측
y_good = np.array([0.95, 0.90, 0.05, 0.10, 0.85])
# 나쁜 예측
y_bad = np.array([0.30, 0.40, 0.80, 0.70, 0.20])

bce_good = bce(y_true, y_good)
bce_bad = bce(y_true, y_bad)

print(f"실제:     {y_true}")
print(f"좋은 예측: {y_good} → BCE = {bce_good:.4f}")
print(f"나쁜 예측: {y_bad}  → BCE = {bce_bad:.4f}")
print(f"→ 잘 맞출수록 BCE가 낮다")

# ── 2. 확신하며 틀리면 페널티 ────────────
print(f"\n[ 확신하며 틀리면 페널티가 극대화 ]")
probs = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
for p in probs:
    loss_when_true = -np.log(p)       # 실제=1일 때
    loss_when_false = -np.log(1 - p)  # 실제=0일 때
    print(f"  p={p:.2f} → 실제1이면 loss={loss_when_true:.3f}, 실제0이면 loss={loss_when_false:.3f}")

# ── 3. CCE 직접 계산 ─────────────────────
print(f"\n[ Categorical Cross-Entropy 직접 계산 ]")

def cce(y_true_onehot, y_pred_softmax):
    epsilon = 1e-15
    y_pred_softmax = np.clip(y_pred_softmax, epsilon, 1.0)
    return -np.sum(y_true_onehot * np.log(y_pred_softmax)) / len(y_true_onehot)

# 3클래스 (정답: 클래스 0)
y_true_oh = np.array([1, 0, 0])
y_good_sm = np.array([0.85, 0.10, 0.05])  # 좋은 예측
y_bad_sm = np.array([0.20, 0.50, 0.30])   # 나쁜 예측

print(f"One-Hot 실제:  {y_true_oh}")
print(f"좋은 Softmax:  {y_good_sm} → CCE = {cce(y_true_oh, y_good_sm):.4f}")
print(f"나쁜 Softmax:  {y_bad_sm}  → CCE = {cce(y_true_oh, y_bad_sm):.4f}")

# ── 4. MSE vs BCE 비교 ──────────────────
print(f"\n[ MSE vs BCE — 왜 분류에 MSE 안 쓰는가 ]")
mse_good = np.mean((y_true - y_good)**2)
mse_bad = np.mean((y_true - y_bad)**2)
print(f"MSE 좋은 예측: {mse_good:.4f}, MSE 나쁜 예측: {mse_bad:.4f}")
print(f"BCE 좋은 예측: {bce_good:.4f}, BCE 나쁜 예측: {bce_bad:.4f}")
print(f"→ BCE가 좋은/나쁜 예측의 차이를 더 크게 만든다")

# ── 5. 시각화 ─────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# BCE: p vs loss (실제=1일 때)
p_range = np.linspace(0.01, 0.99, 100)
loss_true1 = -np.log(p_range)
loss_true0 = -np.log(1 - p_range)
axes[0, 0].plot(p_range, loss_true1, 'b-', linewidth=2, label='Actual=1')
axes[0, 0].plot(p_range, loss_true0, 'r-', linewidth=2, label='Actual=0')
axes[0, 0].set_title('BCE Loss vs Predicted Probability')
axes[0, 0].set_xlabel('Predicted Probability')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()

# 좋은 예측 vs 나쁜 예측
axes[0, 1].bar(['Good Prediction', 'Bad Prediction'], [bce_good, bce_bad],
               color=['green', 'red'], edgecolor='black')
axes[0, 1].set_title('BCE: Good vs Bad')
axes[0, 1].set_ylabel('BCE Loss')

# MSE vs BCE 비교
x_comp = ['MSE Good', 'MSE Bad', 'BCE Good', 'BCE Bad']
y_comp = [mse_good, mse_bad, bce_good, bce_bad]
colors = ['lightgreen', 'lightsalmon', 'green', 'red']
axes[0, 2].bar(x_comp, y_comp, color=colors, edgecolor='black')
axes[0, 2].set_title('MSE vs BCE Comparison')
axes[0, 2].tick_params(axis='x', rotation=30)

# 손실 곡면: MSE로 sigmoid 최적화 (울퉁불퉁)
w_range = np.linspace(-5, 5, 100)
x_data = np.array([1, 2, 3])
y_data = np.array([0, 0, 1])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

mse_surface = [np.mean((y_data - sigmoid(w * x_data))**2) for w in w_range]
bce_surface = []
for w in w_range:
    p = np.clip(sigmoid(w * x_data), 1e-15, 1 - 1e-15)
    bce_val = -np.mean(y_data * np.log(p) + (1 - y_data) * np.log(1 - p))
    bce_surface.append(bce_val)

axes[1, 0].plot(w_range, mse_surface, 'r-', linewidth=2)
axes[1, 0].set_title('MSE Loss Surface (bumpy)')
axes[1, 0].set_xlabel('w')
axes[1, 0].set_ylabel('Loss')

axes[1, 1].plot(w_range, bce_surface, 'b-', linewidth=2)
axes[1, 1].set_title('BCE Loss Surface (smooth)')
axes[1, 1].set_xlabel('w')
axes[1, 1].set_ylabel('Loss')

# 문제 유형별 정리
table_data = [
    ['Regression', 'None', 'MSE'],
    ['Binary Clf', 'Sigmoid', 'BCE'],
    ['Multi Clf', 'Softmax', 'CCE'],
]
axes[1, 2].axis('off')
table = axes[1, 2].table(cellText=table_data,
                          colLabels=['Problem', 'Activation', 'Loss'],
                          loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2)
axes[1, 2].set_title('Problem → Loss Mapping', pad=20)

plt.tight_layout()
plt.savefig('1-11_output.png', dpi=100)
plt.show()

# ── 정리 ──────────────────────────────────
print("\n" + "="*50)
print("핵심 정리:")
print("  회귀 → MSE (loss='mse')")
print("  이진분류 → BCE (loss='binary_crossentropy')")
print("  다중분류 → CCE (loss='categorical_crossentropy')")
print("  BCE는 확신하며 틀리면 페널티 극대화")
print("  분류에 MSE 안 쓰는 이유: 손실 곡면이 울퉁불퉁")
print("="*50)
