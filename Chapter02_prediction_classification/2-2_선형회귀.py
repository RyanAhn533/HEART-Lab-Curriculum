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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.datasets import fetch_california_housing

# ── 1. 단순 선형회귀 (특성 1개) ───────────
print("[ 단순 선형회귀 — 면적 vs 가격 ]")
np.random.seed(42)
area = np.array([20, 25, 30, 35, 40, 45, 50, 55, 60]).reshape(-1, 1)
price = np.array([2.0, 2.3, 2.8, 3.5, 3.9, 4.2, 4.8, 5.3, 5.9])

model_simple = LinearRegression()
model_simple.fit(area, price)

print(f"y = {model_simple.coef_[0]:.4f}x + {model_simple.intercept_:.4f}")
print(f"→ 면적 1평 증가 → 가격 {model_simple.coef_[0]:.2f}억 증가")
print(f"→ 70평 예측: {model_simple.predict([[70]])[0]:.2f}억")
print(f"R² = {model_simple.score(area, price):.4f}")

# ── 2. 다중 선형회귀 (특성 여러 개) ──────
print(f"\n[ 다중 선형회귀 — California Housing ]")
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_multi = LinearRegression()
model_multi.fit(X_train, y_train)
y_pred = model_multi.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R² = {r2:.4f}")
print(f"MSE = {mse:.4f}")
print(f"\n각 특성의 가중치 (w):")
for name, coef in zip(data.feature_names, model_multi.coef_):
    print(f"  {name:12s}: {coef:+.4f}")
print(f"  {'절편(b)':12s}: {model_multi.intercept_:+.4f}")

# ── 3. 잔차 분석 ─────────────────────────
residuals = y_test - y_pred

# ── 4. 시각화 ─────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# 단순 선형회귀
axes[0, 0].scatter(area, price, color='blue', s=50, zorder=5)
x_line = np.linspace(15, 75, 100).reshape(-1, 1)
axes[0, 0].plot(x_line, model_simple.predict(x_line), 'r-', linewidth=2)
axes[0, 0].set_xlabel('Area (pyeong)')
axes[0, 0].set_ylabel('Price (billion)')
axes[0, 0].set_title('Simple Linear Regression')

# 잔차 시각화
for i in range(len(area)):
    pred = model_simple.predict(area[i].reshape(1, -1))[0]
    axes[0, 1].plot([area[i][0], area[i][0]], [price[i], pred], 'g--', alpha=0.7)
axes[0, 1].scatter(area, price, color='blue', s=50, zorder=5)
axes[0, 1].plot(x_line, model_simple.predict(x_line), 'r-', linewidth=2)
axes[0, 1].set_title('Residuals (green lines)')
axes[0, 1].set_xlabel('Area')
axes[0, 1].set_ylabel('Price')

# 다중 회귀: 예측 vs 실제
axes[0, 2].scatter(y_test, y_pred, alpha=0.3, s=5)
axes[0, 2].plot([0, 5], [0, 5], 'r--', linewidth=2, label='Perfect')
axes[0, 2].set_xlabel('Actual Price')
axes[0, 2].set_ylabel('Predicted Price')
axes[0, 2].set_title(f'Actual vs Predicted (R²={r2:.3f})')
axes[0, 2].legend()

# 잔차 분포
axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
axes[1, 0].axvline(0, color='red', linestyle='--')
axes[1, 0].set_title('Residual Distribution')
axes[1, 0].set_xlabel('Residual')

# 가중치 시각화
coef_series = pd.Series(model_multi.coef_, index=data.feature_names).sort_values()
axes[1, 1].barh(coef_series.index, coef_series.values,
                color=['salmon' if v < 0 else 'steelblue' for v in coef_series.values])
axes[1, 1].set_title('Feature Weights (Coefficients)')
axes[1, 1].axvline(0, color='black', linewidth=0.5)

# R² 해석
r2_examples = {'Perfect': 1.0, 'Our Model': r2, 'Average': 0.0, 'Worse': -0.5}
axes[1, 2].barh(list(r2_examples.keys()), list(r2_examples.values()),
                color=['green', 'steelblue', 'gray', 'red'])
axes[1, 2].set_title('R² Score Comparison')
axes[1, 2].set_xlabel('R²')

plt.tight_layout()
plt.savefig('1-7_output.png', dpi=100)
plt.show()

# ── 정리 ──────────────────────────────────
print("\n" + "="*50)
print("핵심 정리:")
print("  선형회귀 = 데이터에 가장 잘 맞는 직선")
print("  y = wx + b (w=기울기, b=절편)")
print("  R² = 모델이 데이터를 얼마나 설명하는가")
print("  잔차 = 실제 - 예측 (작을수록 좋음)")
print("  다중 회귀 = 특성 여러 개 (현실)")
print("="*50)
