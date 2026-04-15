# ============================================
# 1-2. 확률분포 (Probability Distribution)
#
# 왜 배우는가:
#   데이터가 "어떤 모양으로 퍼져 있는가"를 이해해야
#   올바른 전처리와 모델 선택이 가능하다.
#
# 나중에 만나는 곳:
#   → 1-5 스케일링: StandardScaler (정규분포 변환)
#   → 1-4 이상치: Z-score (3σ 밖 = 이상치)
#   → 1-10 로지스틱 회귀: 이항분포 (이진분류 기초)
#
# ▶ 보고 오기: 3B1B "Central Limit Theorem"
#
# Ref: Stanford CS229 W1 / Coursera C1W1
# ============================================

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ── 1. 정규분포 ───────────────────────────
np.random.seed(42)
data_normal = np.random.normal(loc=170, scale=7, size=1000)  # 평균 170, 표준편차 7

mean = np.mean(data_normal)
std = np.std(data_normal)

print("[ 정규분포 — 학생 1000명의 키 (cm) ]")
print(f"평균: {mean:.1f}, 표준편차: {std:.1f}")
print(f"±1σ 범위: {mean-std:.1f} ~ {mean+std:.1f}")

# 68-95-99.7 법칙 확인
within_1s = np.sum((data_normal >= mean - std) & (data_normal <= mean + std)) / len(data_normal) * 100
within_2s = np.sum((data_normal >= mean - 2*std) & (data_normal <= mean + 2*std)) / len(data_normal) * 100
within_3s = np.sum((data_normal >= mean - 3*std) & (data_normal <= mean + 3*std)) / len(data_normal) * 100
print(f"±1σ 안에 {within_1s:.1f}% (이론: 68%)")
print(f"±2σ 안에 {within_2s:.1f}% (이론: 95%)")
print(f"±3σ 안에 {within_3s:.1f}% (이론: 99.7%)")

# ── 2. 이항분포 ───────────────────────────
data_binom = np.random.binomial(n=10, p=0.5, size=1000)  # 동전 10번, 1000회 반복

print(f"\n[ 이항분포 — 동전 10번 던지기 x 1000회 ]")
print(f"앞면 평균 횟수: {np.mean(data_binom):.1f} (이론: 5.0)")

# ── 3. 균등분포 ───────────────────────────
data_uniform = np.random.uniform(low=0, high=100, size=1000)

print(f"\n[ 균등분포 — 0~100 난수 ]")
print(f"평균: {np.mean(data_uniform):.1f} (이론: 50)")

# ── 4. 치우친 분포 (Right Skew) ──────────
data_skewed = np.random.exponential(scale=30, size=1000)  # 소득 같은 분포

skewness = stats.skew(data_skewed)
print(f"\n[ 치우친 분포 — 소득 분포 (지수분포) ]")
print(f"평균: {np.mean(data_skewed):.1f}, 중앙값: {np.median(data_skewed):.1f}")
print(f"왜도(Skewness): {skewness:.2f} (0이면 대칭, 양수면 우로 치우침)")
print(f"→ 평균 > 중앙값 = 우로 치우침 확인")

# ── 5. 로그 변환 효과 ────────────────────
data_log = np.log1p(data_skewed)  # log(1 + x)

print(f"\n[ 로그 변환 후 ]")
print(f"왜도: {stats.skew(data_skewed):.2f} → {stats.skew(data_log):.2f}")
print(f"→ 로그 변환하면 정규분포에 가까워진다")

# ── 6. 시각화 ─────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# 정규분포
axes[0, 0].hist(data_normal, bins=30, density=True, alpha=0.7, edgecolor='black')
x = np.linspace(140, 200, 100)
axes[0, 0].plot(x, stats.norm.pdf(x, mean, std), 'r-', linewidth=2, label='PDF')
axes[0, 0].axvline(mean, color='red', linestyle='--', label=f'Mean={mean:.0f}')
axes[0, 0].axvline(mean-std, color='orange', linestyle=':', alpha=0.7)
axes[0, 0].axvline(mean+std, color='orange', linestyle=':', alpha=0.7, label=f'±1σ')
axes[0, 0].set_title('Normal Distribution')
axes[0, 0].legend(fontsize=8)

# 이항분포
axes[0, 1].hist(data_binom, bins=range(12), density=True, alpha=0.7,
                edgecolor='black', align='left')
axes[0, 1].set_title('Binomial Distribution (n=10, p=0.5)')
axes[0, 1].set_xlabel('# of Heads')

# 균등분포
axes[0, 2].hist(data_uniform, bins=20, density=True, alpha=0.7, edgecolor='black')
axes[0, 2].set_title('Uniform Distribution')

# 치우친 분포 (변환 전)
axes[1, 0].hist(data_skewed, bins=30, density=True, alpha=0.7, edgecolor='black')
axes[1, 0].axvline(np.mean(data_skewed), color='red', linestyle='--', label='Mean')
axes[1, 0].axvline(np.median(data_skewed), color='blue', linestyle='--', label='Median')
axes[1, 0].set_title('Right Skewed (Before)')
axes[1, 0].legend(fontsize=8)

# 로그 변환 후
axes[1, 1].hist(data_log, bins=30, density=True, alpha=0.7, edgecolor='black', color='green')
axes[1, 1].set_title('After Log Transform')

# 중심극한정리 시연
sample_means = [np.mean(np.random.exponential(30, 50)) for _ in range(1000)]
axes[1, 2].hist(sample_means, bins=30, density=True, alpha=0.7, edgecolor='black', color='purple')
axes[1, 2].set_title('Central Limit Theorem\n(sample means of skewed data)')

plt.tight_layout()
plt.savefig('1-2_output.png', dpi=100)
plt.show()

# ── 정리 ──────────────────────────────────
print("\n" + "="*50)
print("핵심 정리:")
print("  정규분포  → 가장 중요, 68-95-99.7 법칙")
print("  이항분포  → 성공/실패 문제 (이진분류 기초)")
print("  치우침    → 평균 vs 중앙값으로 확인")
print("  로그변환  → 치우친 분포를 정규분포에 가깝게")
print("  중심극한정리 → 표본 평균은 항상 정규분포")
print("="*50)
