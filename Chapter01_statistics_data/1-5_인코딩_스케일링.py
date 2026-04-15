# ============================================
# 1-5. 인코딩 / 스케일링
#
# 왜 배우는가:
#   모델은 숫자만 받는다. 범주형은 인코딩, 연속형은 스케일링.
#   스케일링 안 하면 큰 숫자에 모델이 끌려간다.
#
# 나중에 만나는 곳:
#   → 모든 Phase의 #2: 전처리 필수
#   → 1-11 다중분류: One-Hot Encoding
#
# ▶ 보고 오기: Coursera C1W2 "Feature Scaling"
#
# Ref: Google MLCC "Categorical Data" / Coursera C1W2
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler

# ── 1. 인코딩 ─────────────────────────────
print("[ 인코딩 — 범주형을 숫자로 ]")

df = pd.DataFrame({
    'name': ['Kim', 'Lee', 'Park', 'Choi', 'Jung'],
    'department': ['Sales', 'Engineering', 'HR', 'Sales', 'Engineering'],
    'salary': [45000, 62000, 38000, 47000, 58000],
})
print("원본:")
print(df)

# Label Encoding
le = LabelEncoder()
df['dept_label'] = le.fit_transform(df['department'])
print(f"\nLabel Encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# One-Hot Encoding
df_onehot = pd.get_dummies(df[['department']], prefix='dept')
print(f"\nOne-Hot Encoding:")
print(pd.concat([df[['name', 'department']], df_onehot], axis=1))

# ── 2. 스케일링 비교 ─────────────────────
print("\n\n[ 스케일링 — 범위 맞추기 ]")

np.random.seed(42)
data = pd.DataFrame({
    'height': np.random.normal(170, 7, 100),       # 155~185
    'weight': np.random.normal(68, 10, 100),        # 40~100
    'income': np.random.exponential(40000, 100),    # 0~200000 (치우침)
})

# 이상치 추가
data.loc[0, 'income'] = 500000

print("원본 데이터 범위:")
for col in data.columns:
    print(f"  {col}: {data[col].min():.0f} ~ {data[col].max():.0f} (mean={data[col].mean():.0f})")

scalers = {
    'MinMaxScaler': MinMaxScaler(),
    'StandardScaler': StandardScaler(),
    'RobustScaler': RobustScaler(),
}

results = {}
for name, scaler in scalers.items():
    scaled = scaler.fit_transform(data)
    results[name] = pd.DataFrame(scaled, columns=data.columns)
    print(f"\n{name} 후:")
    for col in data.columns:
        print(f"  {col}: {results[name][col].min():.2f} ~ {results[name][col].max():.2f} (mean={results[name][col].mean():.2f})")

# ── 3. fit/transform 분리 (핵심!) ────────
print("\n\n[ fit/transform 분리 — Data Leakage 방지 ]")
from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(data, test_size=0.2, random_state=42)
scaler = StandardScaler()

# 올바른 방법
scaler.fit(X_train)                       # train 기준으로 학습
X_train_scaled = scaler.transform(X_train)  # train 변환
X_test_scaled = scaler.transform(X_test)    # test도 같은 기준

print(f"Train mean (scaled): {X_train_scaled.mean(axis=0).round(2)}")
print(f"Test mean (scaled):  {X_test_scaled.mean(axis=0).round(2)}")
print("→ Train은 0에 가깝고, Test는 약간 다름 = 정상")

# ── 4. 시각화 ─────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# 원본 분포
for i, col in enumerate(data.columns):
    axes[0, i].hist(data[col], bins=20, edgecolor='black', alpha=0.7)
    axes[0, i].set_title(f'Original: {col}')

# 스케일링 비교 (income 열)
colors = ['blue', 'green', 'orange']
for i, (name, result) in enumerate(results.items()):
    axes[1, i].hist(result['income'], bins=20, edgecolor='black', alpha=0.7, color=colors[i])
    axes[1, i].set_title(f'{name}: income')

plt.tight_layout()
plt.savefig('1-5_output.png', dpi=100)
plt.show()

# ── 정리 ──────────────────────────────────
print("\n" + "="*50)
print("핵심 정리:")
print("  Label Encoding → 트리 계열, 순서 있는 범주")
print("  One-Hot → 신경망, 회귀, 순서 없는 범주")
print("  MinMaxScaler → 0~1 범위, 이상치 민감")
print("  StandardScaler → 평균0 표준편차1, 가장 범용적")
print("  RobustScaler → 중앙값/IQR 기준, 이상치에 강함")
print("  fit은 train만! transform은 train+test!")
print("="*50)
