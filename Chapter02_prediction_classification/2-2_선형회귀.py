# ============================================
# 1-7. 선형회귀 (Linear Regression)
#
# 왜 배우는가:
#   ML의 시작 = "직선 하나로 예측".
#   모든 신경망의 기본 구조가 선형회귀의 확장.
#
# 나중에 만나는 곳:
#   → Phase 5: Dense(1) = 선형회귀와 같은 구조
#   → 1-8: MSE = 잔차 제곱의 평균
#
# ▶ 보고 오기: Coursera C1W1 "Linear Regression"
#
# Ref: Stanford CS229 W2 / Coursera C1W1~W2 / Google MLCC
# ============================================

import numpy as np                                    # numpy: 수학 계산 라이브러리 (배열, 행렬 연산 등)
import pandas as pd                                    # pandas: 표(테이블) 형태 데이터를 다루는 라이브러리
import matplotlib.pyplot as plt                        # matplotlib: 그래프를 그리는 라이브러리
from sklearn.linear_model import LinearRegression      # LinearRegression: 직선(y=wx+b)으로 예측하는 모델
from sklearn.model_selection import train_test_split   # train_test_split: 데이터를 학습용/테스트용으로 나누는 함수
from sklearn.metrics import r2_score, mean_squared_error  # r2_score: 모델 설명력 지표, mean_squared_error: 평균 제곱 오차
from sklearn.datasets import fetch_california_housing  # fetch_california_housing: 캘리포니아 집값 예제 데이터셋

# ── 1. 단순 선형회귀 (특성 1개) ───────────
print("[ 단순 선형회귀 — 면적 vs 가격 ]")
np.random.seed(42)                                    # 난수 시드 고정: 실행마다 같은 결과
area = np.array([20, 25, 30, 35, 40, 45, 50, 55, 60]).reshape(-1, 1)  # 면적 데이터 (9개), reshape(-1,1): sklearn이 요구하는 2차원 배열로 변환
price = np.array([2.0, 2.3, 2.8, 3.5, 3.9, 4.2, 4.8, 5.3, 5.9])     # 가격 데이터 (단위: 억원)

model_simple = LinearRegression()                     # 선형회귀 모델 객체 생성 (아직 학습 안 됨)
model_simple.fit(area, price)                         # fit: 데이터로 학습 (최적의 기울기 w와 절편 b를 찾음)

print(f"y = {model_simple.coef_[0]:.4f}x + {model_simple.intercept_:.4f}")  # coef_: 학습된 기울기(w), intercept_: 학습된 절편(b)
print(f"→ 면적 1평 증가 → 가격 {model_simple.coef_[0]:.2f}억 증가")       # 기울기의 실제 의미 해석
print(f"→ 70평 예측: {model_simple.predict([[70]])[0]:.2f}억")             # predict: 학습된 모델로 새 데이터(70평)의 가격 예측
print(f"R² = {model_simple.score(area, price):.4f}")                       # score: R² 값 반환 (1에 가까울수록 데이터를 잘 설명)

# ── 2. 다중 선형회귀 (특성 여러 개) ──────
print(f"\n[ 다중 선형회귀 — California Housing ]")
data = fetch_california_housing()                     # 캘리포니아 집값 데이터 로드 (8개 특성, 2만+ 샘플)
X = pd.DataFrame(data.data, columns=data.feature_names)  # 특성 데이터를 DataFrame으로 변환 (소득, 방 수, 인구 등 8개 열)
y = data.target                                       # 타겟(정답): 집값 (단위: 10만 달러)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 80% 학습용, 20% 테스트용으로 분할, random_state: 분할 결과 고정

model_multi = LinearRegression()                      # 다중 선형회귀 모델 생성
model_multi.fit(X_train, y_train)                     # fit: 8개 특성을 동시에 사용해서 학습 (각 특성마다 가중치 학습)
y_pred = model_multi.predict(X_test)                  # predict: 테스트 데이터로 집값 예측

r2 = r2_score(y_test, y_pred)                         # r2_score: 모델이 데이터 변동의 몇 %를 설명하는지 (1=완벽, 0=평균만큼)
mse = mean_squared_error(y_test, y_pred)              # mean_squared_error: 예측 오차 제곱의 평균 (작을수록 좋음)

print(f"R² = {r2:.4f}")
print(f"MSE = {mse:.4f}")
print(f"\n각 특성의 가중치 (w):")
for name, coef in zip(data.feature_names, model_multi.coef_):  # 각 특성 이름과 학습된 가중치를 짝지어 출력
    print(f"  {name:12s}: {coef:+.4f}")               # +: 양수일 때 +부호 표시 (양수=가격↑, 음수=가격↓)
print(f"  {'절편(b)':12s}: {model_multi.intercept_:+.4f}")  # 모든 특성이 0일 때의 기본 예측값

# ── 3. 잔차 분석 ─────────────────────────
residuals = y_test - y_pred                           # 잔차: 실제값 - 예측값 (0에 가까울수록 좋음)

# ── 4. 시각화 ─────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))      # 2행 3열 = 6개 그래프 영역, 전체 크기 15x9인치

# 단순 선형회귀
axes[0, 0].scatter(area, price, color='blue', s=50, zorder=5)  # scatter: 실제 데이터 점 (s=50: 점 크기, zorder=5: 맨 앞에 표시)
x_line = np.linspace(15, 75, 100).reshape(-1, 1)     # 15~75 범위를 100등분한 점 생성 (예측선 그리기용)
axes[0, 0].plot(x_line, model_simple.predict(x_line), 'r-', linewidth=2)  # 학습된 직선을 빨간색으로 그림
axes[0, 0].set_xlabel('Area (pyeong)')                # x축 라벨: 면적(평)
axes[0, 0].set_ylabel('Price (billion)')              # y축 라벨: 가격(억)
axes[0, 0].set_title('Simple Linear Regression')      # 그래프 제목

# 잔차 시각화
for i in range(len(area)):                            # 각 데이터 점에 대해 반복
    pred = model_simple.predict(area[i].reshape(1, -1))[0]  # 해당 면적의 예측 가격 계산
    axes[0, 1].plot([area[i][0], area[i][0]], [price[i], pred], 'g--', alpha=0.7)  # 실제~예측 사이 초록 점선 = 잔차(오차)
axes[0, 1].scatter(area, price, color='blue', s=50, zorder=5)  # 실제 데이터 점
axes[0, 1].plot(x_line, model_simple.predict(x_line), 'r-', linewidth=2)  # 예측 직선
axes[0, 1].set_title('Residuals (green lines)')       # 제목: 잔차(초록 선)
axes[0, 1].set_xlabel('Area')                         # x축 라벨
axes[0, 1].set_ylabel('Price')                        # y축 라벨

# 다중 회귀: 예측 vs 실제
axes[0, 2].scatter(y_test, y_pred, alpha=0.3, s=5)   # 실제값(x축) vs 예측값(y축) 산점도
axes[0, 2].plot([0, 5], [0, 5], 'r--', linewidth=2, label='Perfect')  # 완벽한 예측일 때의 대각선 (실제=예측)
axes[0, 2].set_xlabel('Actual Price')                 # x축: 실제 가격
axes[0, 2].set_ylabel('Predicted Price')              # y축: 예측 가격
axes[0, 2].set_title(f'Actual vs Predicted (R²={r2:.3f})')  # 제목에 R² 점수 표시
axes[0, 2].legend()                                   # 범례 표시

# 잔차 분포
axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)  # hist: 잔차의 히스토그램 (bins=30: 30개 구간)
axes[1, 0].axvline(0, color='red', linestyle='--')    # 잔차=0 기준선 (이상적으로 잔차가 0 주변에 모여야 함)
axes[1, 0].set_title('Residual Distribution')         # 제목: 잔차 분포
axes[1, 0].set_xlabel('Residual')                     # x축: 잔차 크기

# 가중치 시각화
coef_series = pd.Series(model_multi.coef_, index=data.feature_names).sort_values()  # 특성별 가중치를 Series로 만들어 정렬
axes[1, 1].barh(coef_series.index, coef_series.values,  # barh: 가로 막대 그래프 (각 특성의 가중치)
                color=['salmon' if v < 0 else 'steelblue' for v in coef_series.values])  # 음수=빨강, 양수=파랑
axes[1, 1].set_title('Feature Weights (Coefficients)')  # 제목: 특성별 가중치
axes[1, 1].axvline(0, color='black', linewidth=0.5)  # 가중치=0 기준선

# R² 해석
r2_examples = {'Perfect': 1.0, 'Our Model': r2, 'Average': 0.0, 'Worse': -0.5}  # R² 값 비교용 예시들
axes[1, 2].barh(list(r2_examples.keys()), list(r2_examples.values()),  # 가로 막대로 R² 비교
                color=['green', 'steelblue', 'gray', 'red'])           # 각 막대 색상
axes[1, 2].set_title('R² Score Comparison')           # 제목: R² 비교
axes[1, 2].set_xlabel('R²')                           # x축: R² 값

plt.tight_layout()                                    # 그래프 간 간격 자동 조정
plt.savefig('1-7_output.png', dpi=100)               # 그래프를 이미지 파일로 저장
plt.show()                                            # 화면에 그래프 표시

# ── 정리 ──────────────────────────────────
print("\n" + "="*50)
print("핵심 정리:")
print("  선형회귀 = 데이터에 가장 잘 맞는 직선")
print("  y = wx + b (w=기울기, b=절편)")
print("  R² = 모델이 데이터를 얼마나 설명하는가")
print("  잔차 = 실제 - 예측 (작을수록 좋음)")
print("  다중 회귀 = 특성 여러 개 (현실)")
print("="*50)
