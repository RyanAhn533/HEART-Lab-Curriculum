# ============================================
# 4-2. 활성화 함수 (Activation Functions)
#
# 왜 배우는가:
#   활성화 없으면 레이어 쌓아도 선형.
#   비선형 활성화가 있어야 복잡한 패턴 학습 가능.
#
# 나중에 만나는 곳:
#   → 4-3~4-8: 은닉층 ReLU, 출력층 sigmoid/softmax
#   → tanh: RNN의 기본 활성화 (Chapter 6 예정)
#
# ▶ 보고 오기: 구글 "활성화 함수 종류 비교"
#
# Ref: Coursera C2W2 ReLU Lab / Google MLCC
# ============================================

import numpy as np                              # numpy: 수학 연산 라이브러리 (배열, 행렬 계산에 필수)
import matplotlib.pyplot as plt                  # matplotlib: 그래프/차트를 그리는 시각화 라이브러리

# ── 1. 활성화 함수 정의 ──────────────────
def relu(x):
    """ReLU: 0보다 작으면 0, 크면 그대로 통과. 가장 많이 쓰이는 은닉층 활성화 함수"""
    return np.maximum(0, x)                      # np.maximum: 두 값 중 큰 값을 반환 (음수→0, 양수→그대로)

def sigmoid(x):
    """Sigmoid: 어떤 값이든 0~1 사이로 압축. 이진분류 출력층에 사용"""
    return 1 / (1 + np.exp(-x))                  # np.exp(-x): e의 -x승. 큰 양수→1, 큰 음수→0으로 수렴

def softmax(x):
    """Softmax: 여러 값을 확률로 변환 (합=1). 다중분류 출력층에 사용"""
    exp_x = np.exp(x - np.max(x))                # 최댓값을 빼서 오버플로우(숫자 폭발) 방지
    return exp_x / exp_x.sum()                   # 각 값을 전체 합으로 나눠서 확률로 변환

def linear(x):
    """Linear: 입력을 그대로 출력. 활성화 없는 것과 같음. 회귀 출력층에 사용"""
    return x                                     # 아무 변환 없이 그대로 반환

def tanh(x):
    """Tanh: 어떤 값이든 -1~1 사이로 압축 (0 중심). tanh(x)=2σ(2x)−1. RNN(Chapter 6 예정)의 기본 활성화"""
    return np.tanh(x)                            # numpy 내장 tanh 사용

# ── 2. 시각화 ─────────────────────────────
x = np.linspace(-5, 5, 200)                      # -5~5 사이를 200등분한 배열 생성 (그래프의 x축 데이터)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))  # 2행 3열 = 6개 그래프를 담을 캔버스 생성, 크기 15x9인치

# ReLU
axes[0, 0].plot(x, relu(x), 'b-', linewidth=2)  # ReLU 함수를 파란 실선으로 그림
axes[0, 0].axhline(0, color='gray', linestyle=':', alpha=0.5)  # y=0 수평 보조선 (회색 점선)
axes[0, 0].axvline(0, color='gray', linestyle=':', alpha=0.5)  # x=0 수직 보조선 (회색 점선)
axes[0, 0].set_title('ReLU — Hidden Layers')     # 그래프 제목: ReLU는 은닉층에 사용
axes[0, 0].set_xlabel('x')                       # x축 라벨
axes[0, 0].set_ylabel('ReLU(x)')                 # y축 라벨
axes[0, 0].fill_between(x, relu(x), alpha=0.1, color='blue')  # ReLU 아래 영역을 연한 파란색으로 채움

# Sigmoid vs Tanh (비교)
axes[0, 1].plot(x, sigmoid(x), 'r-', linewidth=2, label='Sigmoid (0~1)')  # Sigmoid 함수를 빨간 실선으로 그림
axes[0, 1].plot(x, tanh(x), 'm--', linewidth=2, label='Tanh (-1~1)')      # Tanh 곡선 추가 — 같은 S자지만 범위가 -1~1, 0 중심
axes[0, 1].axhline(0.5, color='gray', linestyle='--', alpha=0.5)  # y=0.5 기준선 (분류 임계값)
axes[0, 1].axhline(0, color='gray', linestyle=':', alpha=0.5)     # y=0 기준선 (tanh의 중심)
axes[0, 1].set_title('Sigmoid vs Tanh')          # 제목: Sigmoid는 이진분류 출력, Tanh는 RNN 기본 활성화
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('σ(x)')                    # σ(x)는 sigmoid 함수의 수학 표기
axes[0, 1].legend(fontsize=8)                    # 두 곡선 구분용 범례

# Linear (no activation)
axes[0, 2].plot(x, linear(x), 'g-', linewidth=2) # Linear 함수를 초록 실선으로 그림 (y=x 직선)
axes[0, 2].set_title('Linear (No Activation) — Regression Output')  # 제목: 회귀 출력에 사용
axes[0, 2].set_xlabel('x')
axes[0, 2].set_ylabel('x')

# Softmax 예시
logits = np.array([2.0, 1.0, 0.5, -1.0])         # 모델이 출력한 원시 점수 (logits) 4개
probs = softmax(logits)                           # softmax로 확률로 변환 (합=1)
axes[1, 0].bar(['Class 0', 'Class 1', 'Class 2', 'Class 3'], probs,  # 클래스별 확률을 막대그래프로
               color=['steelblue', 'green', 'orange', 'red'], edgecolor='black')
axes[1, 0].set_title(f'Softmax Output (sum={probs.sum():.1f})')  # 확률 합이 1.0임을 보여줌
axes[1, 0].set_ylabel('Probability')              # y축: 확률
for i, p in enumerate(probs):                     # 각 막대 위에 확률 숫자 표시
    axes[1, 0].text(i, p + 0.01, f'{p:.3f}', ha='center', fontsize=10)

# 왜 활성화가 필요한가 — 선형 vs 비선형
np.random.seed(42)                                # 랜덤 시드 고정 (실행할 때마다 같은 결과)
x_data = np.linspace(-3, 3, 100)                  # -3~3 사이 100개 점 생성
y_true = np.sin(x_data)                           # 사인 곡선 = 비선형 패턴 (S자 곡선)

# 선형 모델 (활성화 없음)
from numpy.polynomial import polynomial as P      # 다항식 피팅용 모듈 (여기선 1차=직선 피팅)
coef = np.polyfit(x_data, y_true, 1)              # 1차 다항식(직선) 피팅 → 기울기와 절편 계산
y_linear = np.polyval(coef, x_data)               # 피팅된 직선으로 예측값 생성

axes[1, 1].plot(x_data, y_true, 'b-', linewidth=2, label='True (nonlinear)')   # 실제 비선형 데이터 (파랑)
axes[1, 1].plot(x_data, y_linear, 'r--', linewidth=2, label='Linear (no activation)')  # 선형 모델 예측 (빨간 점선)
axes[1, 1].set_title('Linear Cannot Fit Nonlinear')  # 핵심: 선형은 비선형 패턴을 못 맞춤!
axes[1, 1].legend()                               # 범례 표시

# 정리 표
axes[1, 2].axis('off')                            # 마지막 칸은 그래프 대신 표로 사용 (축 숨김)
table_data = [                                    # 활성화 함수 정리표 데이터
    ['Hidden Layer', 'ReLU', 'Nonlinearity'],                 # 은닉층: ReLU → 비선형성 부여
    ['Output (Regression)', 'None (Linear)', 'Any number'],   # 회귀 출력: 활성화 없음 → 아무 숫자 가능
    ['Output (Binary)', 'Sigmoid', '0~1 probability'],        # 이진분류 출력: Sigmoid → 0~1 확률
    ['Output (Multiclass)', 'Softmax', 'Class probabilities'],# 다중분류 출력: Softmax → 클래스별 확률
]
table = axes[1, 2].table(cellText=table_data,     # 표 생성
                          colLabels=['Location', 'Activation', 'Purpose'],  # 열 제목
                          loc='center', cellLoc='center')     # 표를 가운데 정렬
table.auto_set_font_size(False)                   # 자동 글꼴 크기 비활성화 (수동 설정 위해)
table.set_fontsize(11)                            # 표 글꼴 크기 11pt
table.scale(1, 2)                                 # 표 높이를 2배로 확대 (가독성)
axes[1, 2].set_title('Activation Function Cheat Sheet', pad=20)  # 표 제목, pad=20으로 여백

plt.tight_layout()                                # 그래프 간 겹침 방지 (자동 여백 조정)
plt.savefig('4-2_output.png', dpi=100)            # 결과를 PNG 파일로 저장 (해상도 100dpi)
plt.show()                                        # 화면에 그래프 표시

# ── 정리 ──────────────────────────────────
print("="*50)
print("핵심 정리:")
print("  은닉층 → ReLU (비선형성)")
print("  출력층 회귀 → 없음 (linear)")
print("  출력층 이진분류 → sigmoid (0~1)")
print("  출력층 다중분류 → softmax (확률합=1)")
print("  tanh → -1~1, RNN(Chapter 6 예정)의 기본 활성화")
print("  활성화 없으면 레이어 쌓아도 선형!")
print("="*50)
