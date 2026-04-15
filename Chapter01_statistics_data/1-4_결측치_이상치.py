# ============================================
# 1-4. 결측치 / 이상치 처리
#
# 왜 배우는가:
#   현실 데이터는 비어있거나 이상한 값이 있다.
#   처리 안 하면 모델이 엉뚱한 패턴을 학습한다.
#
# 나중에 만나는 곳:
#   → 모든 Phase의 #2: 전처리 단계
#   → 1-5 스케일링: 이상치 처리 후 스케일링
#
# ▶ 보고 오기: 구글 "pandas 결측치 처리"
#
# Ref: Google MLCC "Numerical Data"
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── 1. 결측치가 있는 데이터 만들기 ────────
np.random.seed(42)
n = 100
df = pd.DataFrame({
    'age': np.random.randint(20, 60, n).astype(float),
    'salary': np.random.normal(50000, 15000, n),
    'department': np.random.choice(['Sales', 'Engineering', 'HR', None], n),
})
# 결측치 삽입
df.loc[np.random.choice(n, 10, replace=False), 'age'] = np.nan
df.loc[np.random.choice(n, 5, replace=False), 'salary'] = np.nan

print("[ 원본 데이터 ]")
print(f"shape: {df.shape}")
print(f"\n결측치 현황:")
print(df.isnull().sum())
print(f"\n처음 10행:")
print(df.head(10))

# ── 2. 결측치 처리 ────────────────────────
df_cleaned = df.copy()

# 연속형: 중앙값으로 대체 (이상치에 강함)
df_cleaned['age'].fillna(df_cleaned['age'].median(), inplace=True)
df_cleaned['salary'].fillna(df_cleaned['salary'].median(), inplace=True)

# 범주형: 최빈값으로 대체
df_cleaned['department'].fillna(df_cleaned['department'].mode()[0], inplace=True)

print(f"\n[ 결측치 처리 후 ]")
print(df_cleaned.isnull().sum())

# ── 3. 이상치 탐지 (IQR 방법) ─────────────
print(f"\n[ 이상치 탐지 — IQR 방법 ]")

def detect_outliers_iqr(series, name):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = series[(series < lower) | (series > upper)]
    print(f"  {name}: Q1={q1:.0f}, Q3={q3:.0f}, IQR={iqr:.0f}")
    print(f"    범위: {lower:.0f} ~ {upper:.0f}")
    print(f"    이상치: {len(outliers)}개")
    return lower, upper

lower_s, upper_s = detect_outliers_iqr(df_cleaned['salary'], 'salary')

# ── 4. 이상치 탐지 (Z-score 방법) ─────────
print(f"\n[ 이상치 탐지 — Z-score 방법 ]")
z_scores = (df_cleaned['salary'] - df_cleaned['salary'].mean()) / df_cleaned['salary'].std()
outliers_z = df_cleaned[np.abs(z_scores) > 3]
print(f"  |Z| > 3인 이상치: {len(outliers_z)}개")

# ── 5. 이상치 처리 (클리핑) ───────────────
df_clipped = df_cleaned.copy()
df_clipped['salary'] = df_clipped['salary'].clip(lower=lower_s, upper=upper_s)

print(f"\n[ 클리핑 처리 후 ]")
print(f"  salary 범위: {df_cleaned['salary'].min():.0f}~{df_cleaned['salary'].max():.0f}")
print(f"  → 클리핑 후: {df_clipped['salary'].min():.0f}~{df_clipped['salary'].max():.0f}")

# ── 6. 시각화 ─────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# 결측치 시각화
missing = df.isnull().sum()
axes[0, 0].bar(missing.index, missing.values, color=['red' if v > 0 else 'green' for v in missing.values])
axes[0, 0].set_title('Missing Values per Column')
axes[0, 0].tick_params(axis='x', rotation=45)

# 처리 전 salary 분포
axes[0, 1].hist(df['salary'].dropna(), bins=20, edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Salary (Before)')

# 처리 후 salary 분포
axes[0, 2].hist(df_cleaned['salary'], bins=20, edgecolor='black', alpha=0.7, color='green')
axes[0, 2].set_title('Salary (After fillna)')

# IQR 이상치 시각화
axes[1, 0].boxplot(df_cleaned['salary'])
axes[1, 0].set_title('Salary Box Plot (IQR)')

# Z-score 분포
axes[1, 1].hist(z_scores, bins=20, edgecolor='black', alpha=0.7, color='purple')
axes[1, 1].axvline(-3, color='red', linestyle='--', label='Z=-3')
axes[1, 1].axvline(3, color='red', linestyle='--', label='Z=+3')
axes[1, 1].set_title('Z-score Distribution')
axes[1, 1].legend()

# 클리핑 전후 비교
axes[1, 2].boxplot([df_cleaned['salary'], df_clipped['salary']],
                    tick_labels=['Before Clip', 'After Clip'])
axes[1, 2].set_title('Before vs After Clipping')

plt.tight_layout()
plt.savefig('1-4_output.png', dpi=100)
plt.show()

# ── 정리 ──────────────────────────────────
print("\n" + "="*50)
print("핵심 정리:")
print("  결측치 → isnull().sum()으로 확인")
print("  연속형 결측 → 중앙값(median) 대체")
print("  범주형 결측 → 최빈값(mode) 대체")
print("  이상치 탐지 → IQR 또는 Z-score")
print("  이상치 처리 → 제거, 클리핑, 또는 유지")
print("="*50)
