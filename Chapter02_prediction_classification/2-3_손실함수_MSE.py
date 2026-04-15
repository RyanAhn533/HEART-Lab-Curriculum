# ============================================
# 1-8. 손실함수 — MSE (Mean Squared Error)
#
# 왜 배우는가:
#   "모델이 얼마나 틀렸는가"를 숫자 하나로 표현.
#   이 숫자를 줄이는 것이 학습(training).
#
# 나중에 만나는 곳:
#   → Phase 5: model.compile(loss='mse')
#   → 1-9: 경사하강법이 이 값을 줄인다
#   → 1-11: 분류용 손실함수 (BCE, CCE)
#
# ▶ 보고 오기: Coursera C1W1 "Cost Function"
#
# Ref: Stanford CS229 W2 / Google MLCC "Loss"
# ============================================

import numpy as np                                    # numpy: 수학 계산 라이브러리 (배열, 제곱, 평균 등)
import matplotlib.pyplot as plt                        # matplotlib: 그래프를 그리는 라이브러리

# ── 1. MSE 직접 계산 ─────────────────────
print("[ MSE 직접 계산 ]")

actual =    np.array([3.0, 2.5, 4.0, 3.5, 5.0])      # 실제값 5개 (정답)
predicted = np.array([2.8, 2.7, 3.5, 3.6, 4.8])      # 예측값 5개 (모델이 예측한 값)
errors = actual - predicted                            # 오차: 실제값 - 예측값 (양수면 과소예측, 음수면 과대예측)

print(f"실제값:  {actual}")
print(f"예측값:  {predicted}")
print(f"오차:    {errors}")
print(f"오차²:   {errors**2}")                         # 오차를 제곱: 음수 오차도 양수로 만들고, 큰 오차에 더 큰 페널티

mse = np.mean(errors**2)                              # MSE: 오차 제곱의 평균 (모델의 전체적인 오차를 하나의 숫자로)
rmse = np.sqrt(mse)                                   # RMSE: MSE에 루트를 씌움 (원래 단위로 해석 가능)
mae = np.mean(np.abs(errors))                         # MAE: 오차 절댓값의 평균 (이상치에 덜 민감)

print(f"\nMSE  = {mse:.4f}")
print(f"RMSE = {rmse:.4f}  ← 원래 단위 (해석 쉬움)")
print(f"MAE  = {mae:.4f}")
print(f"\n→ 평균적으로 {rmse:.2f}만큼 틀린다")

# ── 2. w를 바꾸면 MSE가 어떻게 변하나 ────
print(f"\n[ w를 바꿀 때 MSE 변화 ]")

# 간단한 데이터: y = 2x (정답 w=2)
x = np.array([1, 2, 3, 4, 5], dtype=float)            # 입력값 5개
y = np.array([2, 4, 6, 8, 10], dtype=float)            # 정답: y = 2x

w_values = np.linspace(0, 4, 50)                       # 0~4 사이를 50등분 (다양한 w 후보)
mse_values = []                                        # 각 w에 대한 MSE를 저장할 리스트

for w in w_values:                                     # 각 w 값에 대해 반복
    y_pred = w * x                                     # 현재 w로 예측: y = w * x
    mse_val = np.mean((y - y_pred)**2)                 # 해당 w일 때 MSE 계산
    mse_values.append(mse_val)                         # 결과 저장

best_w = w_values[np.argmin(mse_values)]               # argmin: MSE가 가장 작은 w의 인덱스 → 최적 w
print(f"최적 w = {best_w:.2f} (정답: 2.0)")
print(f"최소 MSE = {min(mse_values):.4f}")

# ── 3. 2D 손실 곡면 (w와 b 동시) ────────
w_range = np.linspace(0, 4, 50)                        # w 후보 값들 (0~4)
b_range = np.linspace(-3, 3, 50)                       # b 후보 값들 (-3~3)
W, B = np.meshgrid(w_range, b_range)                   # meshgrid: w와 b의 모든 조합을 2D 격자로 생성
MSE_surface = np.zeros_like(W)                         # MSE 값을 저장할 빈 배열 (W와 같은 크기)

for i in range(W.shape[0]):                            # 각 행(b 값)에 대해
    for j in range(W.shape[1]):                        # 각 열(w 값)에 대해
        y_pred = W[i, j] * x + B[i, j]                # 해당 w, b로 예측
        MSE_surface[i, j] = np.mean((y - y_pred)**2)   # 해당 w, b 조합의 MSE 계산

# ── 4. 시각화 ─────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))        # 2행 3열 그래프 영역 생성

# 오차 시각화
axes[0, 0].scatter(range(len(actual)), actual, color='blue', s=80, label='Actual', zorder=5)    # 실제값 점
axes[0, 0].scatter(range(len(predicted)), predicted, color='red', s=80, label='Predicted', zorder=5)  # 예측값 점
for i in range(len(actual)):                           # 각 데이터 점마다
    axes[0, 0].plot([i, i], [actual[i], predicted[i]], 'g--', linewidth=2)  # 실제~예측 사이 초록 점선 = 오차
axes[0, 0].set_title('Actual vs Predicted (errors in green)')  # 제목
axes[0, 0].legend()                                   # 범례 표시

# 오차 크기
axes[0, 1].bar(range(len(errors)), errors**2, color='orange', edgecolor='black')  # 각 데이터 점의 오차² 막대 그래프
axes[0, 1].axhline(mse, color='red', linestyle='--', label=f'MSE={mse:.3f}')     # MSE 평균선 (빨간 점선)
axes[0, 1].set_title('Squared Errors')                # 제목: 제곱 오차
axes[0, 1].set_xlabel('Data Point')                   # x축: 데이터 번호
axes[0, 1].set_ylabel('Error²')                       # y축: 오차 제곱 값
axes[0, 1].legend()                                   # 범례

# w vs MSE (1D 손실 곡선)
axes[0, 2].plot(w_values, mse_values, 'b-', linewidth=2)  # w 값에 따른 MSE 변화 곡선
axes[0, 2].scatter([best_w], [min(mse_values)], color='red', s=100, zorder=5, label=f'Best w={best_w:.2f}')  # 최적 w 점 표시
axes[0, 2].set_xlabel('w (weight)')                   # x축: 가중치 w
axes[0, 2].set_ylabel('MSE')                          # y축: MSE 값
axes[0, 2].set_title('Loss Curve: w vs MSE')          # 제목: 손실 곡선
axes[0, 2].legend()                                   # 범례

# 2D 손실 곡면 (contour)
cs = axes[1, 0].contourf(W, B, MSE_surface, levels=20, cmap='viridis')  # contourf: 등고선 색칠 그래프 (MSE가 낮은 곳 = 진한색)
axes[1, 0].scatter([2], [0], color='red', s=100, zorder=5, label='Optimal (w=2, b=0)')  # 최적점 표시
axes[1, 0].set_xlabel('w')                            # x축: 가중치
axes[1, 0].set_ylabel('b')                            # y축: 절편
axes[1, 0].set_title('Loss Surface (contour)')        # 제목: 손실 곡면
axes[1, 0].legend()                                   # 범례
plt.colorbar(cs, ax=axes[1, 0])                       # colorbar: 색상 범례 (MSE 값의 크기)

# MSE vs RMSE vs MAE 비교
metrics = {'MSE': mse, 'RMSE': rmse, 'MAE': mae}     # 세 가지 오차 지표
axes[1, 1].bar(metrics.keys(), metrics.values(), color=['orange', 'green', 'blue'], edgecolor='black')  # 막대 그래프로 비교
axes[1, 1].set_title('MSE vs RMSE vs MAE')            # 제목

# 큰 오차의 영향
errors_small = np.array([0.1, 0.2, 0.1, 0.2, 0.1])   # 모든 오차가 작은 경우
errors_big = np.array([0.1, 0.2, 0.1, 0.2, 2.0])     # 하나만 큰 오차 (이상치)
mse_small = np.mean(errors_small**2)                   # 작은 오차들의 MSE
mse_big = np.mean(errors_big**2)                       # 이상치가 하나 있을 때의 MSE (크게 증가)
axes[1, 2].bar(['All Small Errors', 'One Big Error'], [mse_small, mse_big],
               color=['green', 'red'], edgecolor='black')  # MSE 비교: 이상치 하나가 전체 MSE를 크게 높임
axes[1, 2].set_title(f'MSE: Small={mse_small:.3f} vs Big={mse_big:.3f}')  # 제목에 수치 비교
axes[1, 2].set_ylabel('MSE')                          # y축: MSE 값

plt.tight_layout()                                    # 그래프 간 간격 자동 조정
plt.savefig('1-8_output.png', dpi=100)               # 이미지 파일로 저장
plt.show()                                            # 화면에 표시

# ── 정리 ──────────────────────────────────
print("\n" + "="*50)
print("핵심 정리:")
print("  MSE = 오차²의 평균 (큰 오차에 페널티)")
print("  RMSE = √MSE (원래 단위로 해석)")
print("  MAE = |오차|의 평균 (이상치에 덜 민감)")
print("  w를 바꾸면 MSE가 변한다 → 최소점 찾기 = 학습")
print("  → 이 최소점을 찾는 방법 = 경사하강법 (1-9)")
print("="*50)
