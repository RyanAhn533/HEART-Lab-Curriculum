# ============================================
# 2-2. 활성화 함수 (Activation Functions)
#
# 왜 배우는가:
#   활성화 없으면 레이어 쌓아도 선형.
#   비선형 활성화가 있어야 복잡한 패턴 학습 가능.
#
# 나중에 만나는 곳:
#   → 2-3~2-8: 은닉층 ReLU, 출력층 sigmoid/softmax
#
# ▶ 보고 오기: 구글 "활성화 함수 종류 비교"
#
# Ref: Coursera C2W2 ReLU Lab / Google MLCC
# ============================================

import numpy as np
import matplotlib.pyplot as plt

# ── 1. 활성화 함수 정의 ──────────────────
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

def linear(x):
    return x

# ── 2. 시각화 ─────────────────────────────
x = np.linspace(-5, 5, 200)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# ReLU
axes[0, 0].plot(x, relu(x), 'b-', linewidth=2)
axes[0, 0].axhline(0, color='gray', linestyle=':', alpha=0.5)
axes[0, 0].axvline(0, color='gray', linestyle=':', alpha=0.5)
axes[0, 0].set_title('ReLU — Hidden Layers')
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('ReLU(x)')
axes[0, 0].fill_between(x, relu(x), alpha=0.1, color='blue')

# Sigmoid
axes[0, 1].plot(x, sigmoid(x), 'r-', linewidth=2)
axes[0, 1].axhline(0.5, color='gray', linestyle='--', alpha=0.5)
axes[0, 1].set_title('Sigmoid — Binary Output')
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('σ(x)')

# Linear (no activation)
axes[0, 2].plot(x, linear(x), 'g-', linewidth=2)
axes[0, 2].set_title('Linear (No Activation) — Regression Output')
axes[0, 2].set_xlabel('x')
axes[0, 2].set_ylabel('x')

# Softmax 예시
logits = np.array([2.0, 1.0, 0.5, -1.0])
probs = softmax(logits)
axes[1, 0].bar(['Class 0', 'Class 1', 'Class 2', 'Class 3'], probs,
               color=['steelblue', 'green', 'orange', 'red'], edgecolor='black')
axes[1, 0].set_title(f'Softmax Output (sum={probs.sum():.1f})')
axes[1, 0].set_ylabel('Probability')
for i, p in enumerate(probs):
    axes[1, 0].text(i, p + 0.01, f'{p:.3f}', ha='center', fontsize=10)

# 왜 활성화가 필요한가 — 선형 vs 비선형
np.random.seed(42)
x_data = np.linspace(-3, 3, 100)
y_true = np.sin(x_data)  # 비선형 패턴

# 선형 모델 (활성화 없음)
from numpy.polynomial import polynomial as P
coef = np.polyfit(x_data, y_true, 1)
y_linear = np.polyval(coef, x_data)

axes[1, 1].plot(x_data, y_true, 'b-', linewidth=2, label='True (nonlinear)')
axes[1, 1].plot(x_data, y_linear, 'r--', linewidth=2, label='Linear (no activation)')
axes[1, 1].set_title('Linear Cannot Fit Nonlinear')
axes[1, 1].legend()

# 정리 표
axes[1, 2].axis('off')
table_data = [
    ['Hidden Layer', 'ReLU', 'Nonlinearity'],
    ['Output (Regression)', 'None (Linear)', 'Any number'],
    ['Output (Binary)', 'Sigmoid', '0~1 probability'],
    ['Output (Multiclass)', 'Softmax', 'Class probabilities'],
]
table = axes[1, 2].table(cellText=table_data,
                          colLabels=['Location', 'Activation', 'Purpose'],
                          loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)
axes[1, 2].set_title('Activation Function Cheat Sheet', pad=20)

plt.tight_layout()
plt.savefig('2-2_output.png', dpi=100)
plt.show()

# ── 정리 ──────────────────────────────────
print("="*50)
print("핵심 정리:")
print("  은닉층 → ReLU (비선형성)")
print("  출력층 회귀 → 없음 (linear)")
print("  출력층 이진분류 → sigmoid (0~1)")
print("  출력층 다중분류 → softmax (확률합=1)")
print("  활성화 없으면 레이어 쌓아도 선형!")
print("="*50)
