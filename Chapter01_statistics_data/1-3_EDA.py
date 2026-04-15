# ============================================
# 1-3. 데이터 탐색 (EDA: Exploratory Data Analysis)
#
# 왜 배우는가:
#   모델에 넣기 전에 "눈으로 먼저 보는 것".
#   실무에서 가장 많은 시간을 쓰는 단계.
#
# 나중에 만나는 곳:
#   → 모든 Phase의 #1: 데이터 로드 후 가장 먼저
#   → 1-5 인코딩: 범주형 데이터 변환
#   → 1-6 상관분석: heatmap
#
# ▶ 보고 오기: 구글 "pandas EDA 실습"
#
# Ref: Google MLCC "Numerical/Categorical Data"
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── 1. 데이터 로드 ────────────────────────
# sklearn 내장 데이터셋 (캘리포니아 집값)
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Price'] = data.target

# ── 2. 기본 탐색 (이 6줄을 습관으로) ──────
print("[ 1) shape ]")
print(f"  {df.shape[0]}행, {df.shape[1]}열\n")

print("[ 2) columns & dtypes ]")
print(df.dtypes)

print(f"\n[ 3) head ]")
print(df.head())

print(f"\n[ 4) describe ]")
print(df.describe())

print(f"\n[ 5) 결측치 ]")
print(df.isnull().sum())
print(f"  → 결측치 총 {df.isnull().sum().sum()}개")

print(f"\n[ 6) info ]")
print(df.info())

# ── 3. 시각화 ─────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# 타겟 변수 분포
axes[0, 0].hist(df['Price'], bins=30, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Target: Price Distribution')
axes[0, 0].set_xlabel('Price ($100k)')

# 주요 특성 분포
axes[0, 1].hist(df['MedInc'], bins=30, edgecolor='black', alpha=0.7, color='green')
axes[0, 1].set_title('MedInc (Median Income)')

# 박스플롯 (이상치 확인)
axes[0, 2].boxplot([df['MedInc'], df['HouseAge'], df['AveRooms']],
                    labels=['MedInc', 'HouseAge', 'AveRooms'])
axes[0, 2].set_title('Box Plot (Outlier Check)')

# Scatter (두 변수 관계)
axes[1, 0].scatter(df['MedInc'], df['Price'], alpha=0.3, s=5)
axes[1, 0].set_xlabel('MedInc')
axes[1, 0].set_ylabel('Price')
axes[1, 0].set_title('MedInc vs Price')

# 상관관계 Heatmap
corr = df.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=axes[1, 1], annot_kws={'size': 7})
axes[1, 1].set_title('Correlation Heatmap')

# 상위 상관 특성
corr_price = corr['Price'].drop('Price').sort_values(ascending=False)
axes[1, 2].barh(corr_price.index, corr_price.values, color='steelblue')
axes[1, 2].set_title('Correlation with Price')
axes[1, 2].set_xlabel('Correlation')

plt.tight_layout()
plt.savefig('1-3_output.png', dpi=100)
plt.show()

# ── 4. EDA 체크리스트 결과 ────────────────
print("\n" + "="*50)
print("EDA 체크리스트 결과:")
print(f"  shape: {df.shape}")
print(f"  결측치: {df.isnull().sum().sum()}개")
print(f"  타겟(Price) 평균: {df['Price'].mean():.2f}")
print(f"  타겟(Price) 치우침: {'우로 치우침' if df['Price'].skew() > 0 else '좌로 치우침'}")
print(f"  Price와 가장 상관 높은 특성: {corr_price.index[0]} ({corr_price.values[0]:.3f})")
print("="*50)
