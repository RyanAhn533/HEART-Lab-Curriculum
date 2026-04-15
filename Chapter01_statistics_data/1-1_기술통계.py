# ============================================
# 1-1. 기술통계 (Descriptive Statistics)
#
# 왜 배우는가:
#   데이터를 모델에 넣기 전에 "어떻게 생겼는지" 알아야 한다.
#   기술통계는 데이터를 몇 개의 숫자로 요약하는 방법이다.
#
# 나중에 만나는 곳:
#   → 1-5 스케일링: StandardScaler = (x - mean) / std
#   → 1-4 이상치 처리: IQR 기준 제거
#   → 1-3 EDA: pandas describe()
#
# ▶ 보고 오기: 구글 "기술통계 파이썬"
#
# Ref: Stanford CS229 W1 / Google MLCC "Numerical Data"
# ============================================

import numpy as np
import matplotlib.pyplot as plt

# ── 1. 데이터 준비 ────────────────────────
# 학생 30명의 시험 점수 (가상)
np.random.seed(42)
scores = np.random.normal(loc=70, scale=12, size=30).astype(int)
scores = np.append(scores, [15, 98])  # 이상치 추가
print(f"데이터: {sorted(scores)}")
print(f"데이터 개수: {len(scores)}")

# ── 2. 중심 경향 ──────────────────────────
mean = np.mean(scores)
median = np.median(scores)
print(f"\n[ 중심 경향 ]")
print(f"평균(Mean):   {mean:.1f}")
print(f"중앙값(Median): {median:.1f}")
print(f"→ 평균과 중앙값이 다르면 이상치가 있을 수 있다")

# ── 3. 산포도 ─────────────────────────────
var = np.var(scores)
std = np.std(scores)
print(f"\n[ 산포도 ]")
print(f"분산(Variance):       {var:.1f}")
print(f"표준편차(Std Dev):     {std:.1f}")
print(f"최소~최대: {np.min(scores)} ~ {np.max(scores)}")
print(f"→ 데이터가 평균에서 약 {std:.0f}점 정도 퍼져있다")

# ── 4. 사분위수 / IQR ────────────────────
q1 = np.percentile(scores, 25)
q3 = np.percentile(scores, 75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

print(f"\n[ 사분위수 ]")
print(f"Q1 (25%): {q1:.1f}")
print(f"Q2 (50%): {median:.1f}")
print(f"Q3 (75%): {q3:.1f}")
print(f"IQR:      {iqr:.1f}")
print(f"이상치 기준: < {lower:.1f} 또는 > {upper:.1f}")

outliers = scores[(scores < lower) | (scores > upper)]
print(f"이상치: {outliers}")

# ── 5. 시각화 ─────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# 히스토그램
axes[0].hist(scores, bins=10, edgecolor='black', alpha=0.7)
axes[0].axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean={mean:.1f}')
axes[0].axvline(median, color='blue', linestyle='--', linewidth=2, label=f'Median={median:.1f}')
axes[0].set_title('Score Distribution (Histogram)')
axes[0].set_xlabel('Score')
axes[0].set_ylabel('Frequency')
axes[0].legend()

# 박스플롯
bp = axes[1].boxplot(scores, vert=True, patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')
axes[1].set_title('Score Distribution (Box Plot)')
axes[1].set_ylabel('Score')

# 이상치 표시
normal = scores[(scores >= lower) & (scores <= upper)]
axes[2].scatter(range(len(normal)), sorted(normal), color='blue', alpha=0.6, label='Normal')
axes[2].scatter(range(len(normal), len(normal)+len(outliers)), sorted(outliers),
                color='red', s=100, marker='x', linewidths=2, label='Outlier')
axes[2].axhline(lower, color='red', linestyle=':', alpha=0.5, label=f'Lower={lower:.0f}')
axes[2].axhline(upper, color='red', linestyle=':', alpha=0.5, label=f'Upper={upper:.0f}')
axes[2].set_title('Normal vs Outlier')
axes[2].set_ylabel('Score')
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig('1-1_output.png', dpi=100)
plt.show()

# ── 6. pandas로 한 번에 ──────────────────
import pandas as pd
df = pd.DataFrame({'score': scores})
print("\n[ pandas describe() — 실무에서는 이걸 쓴다 ]")
print(df.describe())

# ── 정리 ──────────────────────────────────
print("\n" + "="*50)
print("핵심 정리:")
print("  평균/중앙값 → 데이터의 중심")
print("  표준편차    → 데이터의 퍼짐 정도")
print("  IQR        → 이상치 판별 기준")
print("  describe()  → 실무에서 가장 먼저 치는 명령어")
print("="*50)
