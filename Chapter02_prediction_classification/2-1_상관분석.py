# ============================================
# 1-6. 상관분석 (Correlation Analysis)
#
# 왜 배우는가:
#   "이 특성(X)이 결과(Y)와 관련이 있는가?"
#   관련 없는 특성을 넣으면 모델 성능이 떨어진다.
#
# 나중에 만나는 곳:
#   → 1-7 선형회귀: 상관 높은 특성으로 예측
#   → Phase 11 Feature Importance: 특성 선택
#
# ▶ 보고 오기: StatQuest "Correlation"
#
# Ref: Stanford CS229 W1 / Google MLCC
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# ── 1. 데이터 로드 ────────────────────────
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Price'] = data.target

# ── 2. 상관계수 계산 ──────────────────────
corr_matrix = df.corr()
corr_with_price = corr_matrix['Price'].drop('Price').sort_values(ascending=False)

print("[ Price와의 상관계수 (Pearson) ]")
for col, val in corr_with_price.items():
    strength = "강함" if abs(val) > 0.4 else "약함" if abs(val) > 0.2 else "거의 없음"
    print(f"  {col:12s}: {val:+.3f}  ({strength})")

# ── 3. 상관 ≠ 인과 예시 ──────────────────
print("\n[ 주의: 상관 ≠ 인과 ]")
print("  Latitude와 Price 상관 = -0.14")
print("  → 위도가 가격을 결정하는 게 아니라,")
print("  → 특정 위도(LA, SF)에 비싼 집이 많은 것")

# ── 4. 시각화 ─────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# 상관 heatmap
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=axes[0, 0], annot_kws={'size': 7})
axes[0, 0].set_title('Correlation Heatmap')

# 상관계수 바 차트
colors = ['steelblue' if v > 0 else 'salmon' for v in corr_with_price.values]
axes[0, 1].barh(corr_with_price.index, corr_with_price.values, color=colors)
axes[0, 1].set_title('Correlation with Price')
axes[0, 1].axvline(0, color='black', linewidth=0.5)

# 강한 양의 상관: MedInc vs Price
axes[0, 2].scatter(df['MedInc'], df['Price'], alpha=0.2, s=3)
z = np.polyfit(df['MedInc'], df['Price'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['MedInc'].min(), df['MedInc'].max(), 100)
axes[0, 2].plot(x_line, p(x_line), 'r-', linewidth=2, label=f'r={corr_matrix.loc["MedInc","Price"]:.3f}')
axes[0, 2].set_xlabel('MedInc')
axes[0, 2].set_ylabel('Price')
axes[0, 2].set_title('Strong Positive Correlation')
axes[0, 2].legend()

# 약한 상관: Population vs Price
axes[1, 0].scatter(df['Population'], df['Price'], alpha=0.2, s=3)
axes[1, 0].set_xlabel('Population')
axes[1, 0].set_ylabel('Price')
axes[1, 0].set_title(f'Weak Correlation (r={corr_matrix.loc["Population","Price"]:.3f})')

# 음의 상관: Latitude vs Price
axes[1, 1].scatter(df['Latitude'], df['Price'], alpha=0.2, s=3)
axes[1, 1].set_xlabel('Latitude')
axes[1, 1].set_ylabel('Price')
axes[1, 1].set_title(f'Negative Correlation (r={corr_matrix.loc["Latitude","Price"]:.3f})')

# 상관 없음 예시: 랜덤
np.random.seed(42)
x_rand = np.random.randn(500)
y_rand = np.random.randn(500)
axes[1, 2].scatter(x_rand, y_rand, alpha=0.3, s=10)
r_rand = np.corrcoef(x_rand, y_rand)[0, 1]
axes[1, 2].set_title(f'No Correlation (r={r_rand:.3f})')

plt.tight_layout()
plt.savefig('1-6_output.png', dpi=100)
plt.show()

# ── 정리 ──────────────────────────────────
print("\n" + "="*50)
print("핵심 정리:")
print("  상관계수 = -1 ~ +1 (관계의 방향과 강도)")
print("  |r| > 0.4 → 쓸만한 특성")
print("  상관 ≠ 인과 (항상 주의)")
print("  scatter + 추세선 = 다음에 배울 선형회귀")
print("="*50)
