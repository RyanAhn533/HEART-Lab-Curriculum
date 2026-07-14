# ============================================
# 2-4. 경사하강법 (Gradient Descent)
#
# 왜 배우는가:
#   손실(MSE)을 줄이는 방법.
#   모든 딥러닝 학습의 핵심 원리.
#
# 나중에 만나는 곳:
#   → Chapter 4: optimizer='adam'
#   → 후속 챕터: ReduceLROnPlateau
#
# ▶ 보고 오기: 3B1B "Gradient descent" (필수!)
#
# Ref: Stanford CS229 W2 / Coursera C1W1
# ============================================

import numpy as np                                    # numpy: 수학 계산 라이브러리 (배열, 합, 평균 등)
import matplotlib.pyplot as plt                        # matplotlib: 그래프를 그리는 라이브러리

# ── 1. 데이터 (y = 2x, 정답 w=2) ────────
x = np.array([1, 2, 3, 4, 5], dtype=float)            # 입력값 5개
y = np.array([2, 4, 6, 8, 10], dtype=float)            # 정답: y = 2x (기울기=2가 정답)

# ── 2. 경사하강법 직접 구현 ──────────────
def gradient_descent(x, y, lr, epochs):                # lr: 학습률(한 걸음 크기), epochs: 반복 횟수
    w = 0.0                                            # 초기 가중치 (0에서 시작, 정답 2를 찾아야 함)
    b = 0.0                                            # 초기 편향 (절편, 0에서 시작)
    n = len(x)                                         # 데이터 개수
    history = {'w': [], 'b': [], 'mse': []}            # 학습 과정을 기록할 딕셔너리

    for epoch in range(epochs):                        # epochs 횟수만큼 반복 학습
        # 순전파: 예측
        y_pred = w * x + b                             # 현재 w, b로 예측값 계산

        # 손실 계산
        mse = np.mean((y - y_pred)**2)                 # MSE: 현재 모델이 얼마나 틀렸는지

        # 기울기(gradient) 계산
        dw = -(2/n) * np.sum(x * (y - y_pred))         # dw: MSE를 w로 미분한 값 (w를 어느 방향으로 바꿔야 하는지)
        db = -(2/n) * np.sum(y - y_pred)               # db: MSE를 b로 미분한 값 (b를 어느 방향으로 바꿔야 하는지)

        # 가중치 업데이트
        w = w - lr * dw                                # w를 기울기 반대 방향으로 lr만큼 이동 (핵심 공식!)
        b = b - lr * db                                # b도 마찬가지로 업데이트

        history['w'].append(w)                         # 현재 w 기록
        history['b'].append(b)                         # 현재 b 기록
        history['mse'].append(mse)                     # 현재 MSE 기록

        if epoch % 20 == 0 or epoch == epochs-1:       # 20번마다 또는 마지막에 진행 상황 출력
            print(f"  Epoch {epoch:3d}: w={w:.4f}, b={b:.4f}, MSE={mse:.4f}")

    return w, b, history                               # 최종 w, b와 학습 기록 반환

# ── 3. 적당한 lr ─────────────────────────
print("[ lr=0.01 — 적당한 학습률 ]")
w_good, b_good, hist_good = gradient_descent(x, y, lr=0.01, epochs=200)  # 적당한 학습률로 200번 반복

# ── 4. 너무 큰 lr ────────────────────────
print(f"\n[ lr=0.1 — 너무 큰 학습률 ]")
w_big, b_big, hist_big = gradient_descent(x, y, lr=0.1, epochs=20)  # 큰 학습률 → 발산 위험

# ── 5. 너무 작은 lr ──────────────────────
print(f"\n[ lr=0.001 — 너무 작은 학습률 ]")
w_small, b_small, hist_small = gradient_descent(x, y, lr=0.001, epochs=200)  # 작은 학습률 → 수렴이 느림

# ── 6. 시각화 ─────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))        # 2행 3열 그래프 영역

# 학습 곡선 (MSE vs Epoch) — 적당한 lr
axes[0, 0].plot(hist_good['mse'], 'b-', linewidth=2)  # MSE가 에폭마다 줄어드는 과정 (파란색)
axes[0, 0].set_title('lr=0.01 (Good)')                 # 제목: 적당한 학습률
axes[0, 0].set_xlabel('Epoch')                         # x축: 반복 횟수
axes[0, 0].set_ylabel('MSE')                           # y축: 손실값
axes[0, 0].set_ylim(bottom=0)                          # y축 최솟값 0으로 고정

# 학습 곡선 — 큰 lr
axes[0, 1].plot(hist_big['mse'], 'r-', linewidth=2)    # 큰 lr: MSE가 진동하거나 발산 (빨간색)
axes[0, 1].set_title('lr=0.1 (Too Big)')               # 제목: 너무 큰 학습률
axes[0, 1].set_xlabel('Epoch')                         # x축
axes[0, 1].set_ylabel('MSE')                           # y축

# 학습 곡선 — 작은 lr
axes[0, 2].plot(hist_small['mse'], 'g-', linewidth=2)  # 작은 lr: MSE가 천천히 줄어듦 (초록색)
axes[0, 2].set_title('lr=0.001 (Too Small)')           # 제목: 너무 작은 학습률
axes[0, 2].set_xlabel('Epoch')                         # x축
axes[0, 2].set_ylabel('MSE')                           # y축
axes[0, 2].set_ylim(bottom=0)                          # y축 최솟값 0

# w의 변화 과정
axes[1, 0].plot(hist_good['w'], 'b-', label='lr=0.01')    # 적당한 lr: w가 빠르게 2에 수렴
axes[1, 0].plot(hist_small['w'], 'g-', label='lr=0.001')  # 작은 lr: w가 천천히 2에 접근
axes[1, 0].axhline(2.0, color='red', linestyle='--', label='Optimal w=2')  # 정답 w=2 기준선
axes[1, 0].set_title('w Convergence')                  # 제목: w 수렴 과정
axes[1, 0].set_xlabel('Epoch')                         # x축: 반복 횟수
axes[1, 0].set_ylabel('w')                             # y축: 가중치 값
axes[1, 0].legend()                                    # 범례 표시

# 손실 곡면 위에서 경사하강법 경로
w_range = np.linspace(-0.5, 4, 100)                    # w 범위: -0.5~4
mse_curve = [np.mean((y - w_val * x)**2) for w_val in w_range]  # 각 w에 대한 MSE 계산 (U자 곡선)
axes[1, 1].plot(w_range, mse_curve, 'k-', linewidth=2)  # 손실 곡선 (검은색)
axes[1, 1].plot(hist_good['w'], hist_good['mse'], 'bo-', markersize=3, alpha=0.5, label='GD path')  # 경사하강법이 이동한 경로
axes[1, 1].scatter([2], [0], color='red', s=100, zorder=5, label='Optimal')  # 최적점 (w=2, MSE=0)
axes[1, 1].set_title('GD Path on Loss Curve')         # 제목: 손실 곡선 위의 경사하강법 경로
axes[1, 1].set_xlabel('w')                             # x축: 가중치
axes[1, 1].set_ylabel('MSE')                           # y축: 손실값
axes[1, 1].legend()                                    # 범례

# 최종 결과 비교
axes[1, 2].scatter(x, y, color='blue', s=80, label='Data', zorder=5)  # 원본 데이터 점
x_line = np.linspace(0, 6, 50)                         # 예측선 그리기용 x 범위
axes[1, 2].plot(x_line, w_good * x_line + b_good, 'b-', linewidth=2,
                label=f'GD: y={w_good:.2f}x+{b_good:.2f}')  # 경사하강법으로 찾은 직선
axes[1, 2].plot(x_line, 2 * x_line, 'r--', linewidth=1, label='y=2x (answer)')  # 정답 직선
axes[1, 2].set_title('Final Result')                   # 제목: 최종 결과
axes[1, 2].set_xlabel('x')                             # x축
axes[1, 2].set_ylabel('y')                             # y축
axes[1, 2].legend()                                    # 범례

plt.tight_layout()                                     # 그래프 간 간격 자동 조정
plt.savefig('2-4_output.png', dpi=100)                # 이미지 파일로 저장
plt.show()                                             # 화면에 표시

# ── 정리 ──────────────────────────────────
print("\n" + "="*50)
print("핵심 정리:")
print(f"  경사하강법 = 기울기 방향으로 한 발짝씩 이동")
print(f"  w_new = w_old - lr * gradient")
print(f"  lr 너무 크면 → 발산, 너무 작으면 → 느림")
print(f"  최종 결과: w={w_good:.4f} (정답: 2.0)")
print(f"  → 이게 optimizer='adam'이 하는 일")
print("="*50)
