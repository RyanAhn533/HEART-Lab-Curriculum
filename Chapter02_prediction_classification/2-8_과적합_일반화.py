# ============================================
# 1-13. 과적합과 일반화 (Overfitting & Generalization)
#
# 왜 배우는가:
#   훈련 데이터에서 잘 되는데 새 데이터에서 안 되는 이유.
#   ML에서 가장 흔하고 중요한 문제.
#
# 나중에 만나는 곳:
#   → Phase 6: EarlyStopping, Dropout
#   → Phase 11: KFold, cross_val_score
#
# ▶ 보고 오기: Coursera C1W3 "Overfitting"
#
# Ref: Stanford CS229 W4 / Google MLCC "Generalization"
# ============================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

# ── 1. 과적합 시연 데이터 ────────────────
np.random.seed(42)
n = 30
X = np.sort(np.random.uniform(0, 1, n)).reshape(-1, 1)
y = np.sin(2 * np.pi * X).ravel() + np.random.normal(0, 0.2, n)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_plot = np.linspace(0, 1, 200).reshape(-1, 1)

# ── 2. 복잡도별 모델 비교 ────────────────
degrees = [1, 4, 15]  # 과소적합, 적절, 과적합
models = {}

print("[ 모델 복잡도별 성능 비교 ]")
print(f"{'Degree':>8} {'Train MSE':>12} {'Test MSE':>12} {'상태':>10}")
print("-" * 45)

for d in degrees:
    pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=d)),
        ('lr', LinearRegression())
    ])
    pipe.fit(X_train, y_train)

    train_mse = mean_squared_error(y_train, pipe.predict(X_train))
    test_mse = mean_squared_error(y_test, pipe.predict(X_test))

    if d == 1:
        status = "과소적합"
    elif d == 4:
        status = "적절"
    else:
        status = "과적합"

    print(f"{d:>8} {train_mse:>12.4f} {test_mse:>12.4f} {status:>10}")
    models[d] = {'pipe': pipe, 'train_mse': train_mse, 'test_mse': test_mse}

# ── 3. 정규화 (Ridge) 효과 ───────────────
print(f"\n[ 정규화 (Ridge) — 과적합 방지 ]")
pipe_overfit = Pipeline([
    ('poly', PolynomialFeatures(degree=15)),
    ('lr', LinearRegression())
])
pipe_ridge = Pipeline([
    ('poly', PolynomialFeatures(degree=15)),
    ('ridge', Ridge(alpha=0.1))
])

pipe_overfit.fit(X_train, y_train)
pipe_ridge.fit(X_train, y_train)

test_overfit = mean_squared_error(y_test, pipe_overfit.predict(X_test))
test_ridge = mean_squared_error(y_test, pipe_ridge.predict(X_test))
print(f"  Degree=15 (정규화 없음): Test MSE = {test_overfit:.4f}")
print(f"  Degree=15 + Ridge:      Test MSE = {test_ridge:.4f}")
print(f"  → 정규화로 과적합 완화")

# ── 4. 교차검증 ──────────────────────────
print(f"\n[ 교차검증 (5-Fold) ]")
for d in degrees:
    pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=d)),
        ('lr', LinearRegression())
    ])
    scores = cross_val_score(pipe, X, y, cv=5, scoring='neg_mean_squared_error')
    mean_mse = -scores.mean()
    std_mse = scores.std()
    print(f"  Degree={d:2d}: MSE = {mean_mse:.4f} (±{std_mse:.4f})")

# ── 5. 시각화 ─────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# 과소적합 / 적절 / 과적합
titles = {1: 'Underfitting (d=1)', 4: 'Good Fit (d=4)', 15: 'Overfitting (d=15)'}
for i, d in enumerate(degrees):
    pipe = models[d]['pipe']
    y_plot = pipe.predict(X_plot)

    axes[0, i].scatter(X_train, y_train, color='blue', s=30, label='Train', zorder=5)
    axes[0, i].scatter(X_test, y_test, color='red', s=30, label='Test', zorder=5)
    axes[0, i].plot(X_plot, y_plot, 'g-', linewidth=2, label=f'd={d}')
    axes[0, i].plot(X_plot, np.sin(2 * np.pi * X_plot), 'k--', alpha=0.3, label='True')
    axes[0, i].set_title(titles[d])
    axes[0, i].set_ylim(-2, 2)
    axes[0, i].legend(fontsize=7)

# 복잡도 vs 성능
degrees_range = range(1, 16)
train_errors = []
test_errors = []
for d in degrees_range:
    pipe = Pipeline([('poly', PolynomialFeatures(degree=d)), ('lr', LinearRegression())])
    pipe.fit(X_train, y_train)
    train_errors.append(mean_squared_error(y_train, pipe.predict(X_train)))
    test_errors.append(mean_squared_error(y_test, pipe.predict(X_test)))

axes[1, 0].plot(list(degrees_range), train_errors, 'b-o', markersize=4, label='Train')
axes[1, 0].plot(list(degrees_range), test_errors, 'r-o', markersize=4, label='Test')
axes[1, 0].set_xlabel('Polynomial Degree')
axes[1, 0].set_ylabel('MSE')
axes[1, 0].set_title('Complexity vs Error')
axes[1, 0].set_ylim(0, min(2, max(test_errors)))
axes[1, 0].axvline(4, color='green', linestyle='--', alpha=0.5, label='Sweet Spot')
axes[1, 0].legend()

# 정규화 효과
axes[1, 1].scatter(X_train, y_train, color='blue', s=30, zorder=5)
axes[1, 1].scatter(X_test, y_test, color='red', s=30, zorder=5)
axes[1, 1].plot(X_plot, pipe_overfit.predict(X_plot), 'orange', linewidth=2, label='No Regularization')
axes[1, 1].plot(X_plot, pipe_ridge.predict(X_plot), 'green', linewidth=2, label='Ridge (L2)')
axes[1, 1].set_ylim(-2, 2)
axes[1, 1].set_title('Regularization Effect (d=15)')
axes[1, 1].legend()

# Bias-Variance 직관
labels = ['Underfitting', 'Sweet Spot', 'Overfitting']
bias_vals = [0.8, 0.2, 0.05]
var_vals = [0.05, 0.2, 0.8]
total = [b + v for b, v in zip(bias_vals, var_vals)]
x_pos = range(len(labels))
axes[1, 2].bar(x_pos, bias_vals, 0.35, label='Bias', color='steelblue')
axes[1, 2].bar(x_pos, var_vals, 0.35, bottom=bias_vals, label='Variance', color='salmon')
axes[1, 2].plot(x_pos, total, 'ko-', label='Total Error')
axes[1, 2].set_xticks(x_pos)
axes[1, 2].set_xticklabels(labels)
axes[1, 2].set_title('Bias-Variance Tradeoff')
axes[1, 2].legend()

plt.tight_layout()
plt.savefig('1-13_output.png', dpi=100)
plt.show()

# ── 정리 ──────────────────────────────────
print("\n" + "="*50)
print("핵심 정리:")
print("  과소적합 = 모델 너무 단순 → 레이어/특성 추가")
print("  과적합   = 모델 너무 복잡 → 정규화/데이터 추가")
print("  train/test 분할 = 반드시 나눠서 평가")
print("  교차검증 = K번 반복해서 안정적 평가")
print("  정규화 = 가중치를 크게 만들지 않는 제약")
print("="*50)
print("\n★ Phase 1 완료! → 체크포인트 시험 1")
