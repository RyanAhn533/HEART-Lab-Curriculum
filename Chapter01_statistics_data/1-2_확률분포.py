# ============================================
# 1-2. 확률분포 (Probability Distribution)
#
# 왜 배우는가:
#   데이터가 "어떤 모양으로 퍼져 있는가"를 이해해야
#   올바른 전처리와 모델 선택이 가능하다.
#
# 나중에 만나는 곳:
#   → 1-6 스케일링: StandardScaler (정규분포 변환)
#   → 1-5 이상치: Z-score (3σ 밖 = 이상치)
#   → 2-5 로지스틱 회귀: 이항분포 (이진분류 기초)
#
# ▶ 보고 오기: 3B1B "Central Limit Theorem"
#
# Ref: Stanford CS229 W1 / Coursera C1W1
# ============================================

import numpy as np                # numpy: 수학 계산 라이브러리 (배열, 평균, 표준편차 등)
import matplotlib.pyplot as plt   # matplotlib: 그래프/차트를 그리는 라이브러리
from scipy import stats           # scipy.stats: 통계 함수 모음 (왜도, 확률밀도함수 등)

# ── 1. 정규분포 ───────────────────────────
np.random.seed(42)                # 랜덤 시드 고정 — 실행할 때마다 같은 결과가 나오게
data_normal = np.random.normal(loc=170, scale=7, size=1000)  # loc=평균 170cm, scale=표준편차 7cm, size=1000명 생성

mean = np.mean(data_normal)       # mean: 1000개 데이터의 평균값 계산
std = np.std(data_normal)         # std: 1000개 데이터의 표준편차 계산 (데이터가 평균에서 얼마나 퍼져있는지)

print("[ 정규분포 — 학생 1000명의 키 (cm) ]")
print(f"평균: {mean:.1f}, 표준편차: {std:.1f}")         # :.1f → 소수점 첫째 자리까지 출력
print(f"±1σ 범위: {mean-std:.1f} ~ {mean+std:.1f}")    # σ(시그마) = 표준편차. 평균 ± 1σ 범위 출력

# 68-95-99.7 법칙 확인 — 정규분포에서 데이터가 각 σ 범위 안에 몇 % 들어가는지 검증
within_1s = np.sum((data_normal >= mean - std) & (data_normal <= mean + std)) / len(data_normal) * 100
# ↑ 평균 ± 1σ 범위 안에 있는 데이터 개수를 세고, 전체 대비 비율(%)로 계산
within_2s = np.sum((data_normal >= mean - 2*std) & (data_normal <= mean + 2*std)) / len(data_normal) * 100
# ↑ 평균 ± 2σ 범위 안에 있는 비율
within_3s = np.sum((data_normal >= mean - 3*std) & (data_normal <= mean + 3*std)) / len(data_normal) * 100
# ↑ 평균 ± 3σ 범위 안에 있는 비율
print(f"±1σ 안에 {within_1s:.1f}% (이론: 68%)")       # 이론적으로 68%가 이 범위 안에 있어야 함
print(f"±2σ 안에 {within_2s:.1f}% (이론: 95%)")       # 이론적으로 95%
print(f"±3σ 안에 {within_3s:.1f}% (이론: 99.7%)")     # 이론적으로 99.7%

# ── 2. 이항분포 ───────────────────────────
data_binom = np.random.binomial(n=10, p=0.5, size=1000)  # n=시행 횟수 10번, p=성공 확률 0.5, size=1000회 반복
# ↑ "동전 10번 던져서 앞면이 몇 번 나오는지"를 1000번 반복한 결과

print(f"\n[ 이항분포 — 동전 10번 던지기 x 1000회 ]")
print(f"앞면 평균 횟수: {np.mean(data_binom):.1f} (이론: 5.0)")  # 이론적 평균 = n × p = 10 × 0.5 = 5

# ── 3. 균등분포 ───────────────────────────
data_uniform = np.random.uniform(low=0, high=100, size=1000)  # low=최솟값 0, high=최댓값 100, 모든 값이 동일한 확률로 생성

print(f"\n[ 균등분포 — 0~100 난수 ]")
print(f"평균: {np.mean(data_uniform):.1f} (이론: 50)")  # 0~100 균등분포의 이론적 평균 = (0+100)/2 = 50

# ── 4. 치우친 분포 (Right Skew) ──────────
data_skewed = np.random.exponential(scale=30, size=1000)  # 지수분포: scale=평균 30. 소득 분포처럼 오른쪽으로 긴 꼬리를 가진 분포
# ↑ 대부분 낮은 값에 몰리고, 일부가 아주 큰 값을 가지는 형태

skewness = stats.skew(data_skewed)  # 왜도 계산: 분포가 얼마나 치우쳤는지 수치화 (0=대칭, 양수=오른쪽 꼬리)
print(f"\n[ 치우친 분포 — 소득 분포 (지수분포) ]")
print(f"평균: {np.mean(data_skewed):.1f}, 중앙값: {np.median(data_skewed):.1f}")  # 치우친 분포에서는 평균 > 중앙값
print(f"왜도(Skewness): {skewness:.2f} (0이면 대칭, 양수면 우로 치우침)")
print(f"→ 평균 > 중앙값 = 우로 치우침 확인")

# ── 5. 로그 변환 효과 ────────────────────
data_log = np.log1p(data_skewed)  # log1p = log(1 + x). 치우친 분포를 정규분포에 가깝게 변환
# ↑ log1p를 쓰는 이유: log(0)은 -∞이므로 1을 더해서 방지

print(f"\n[ 로그 변환 후 ]")
print(f"왜도: {stats.skew(data_skewed):.2f} → {stats.skew(data_log):.2f}")  # 변환 전후 왜도 비교
print(f"→ 로그 변환하면 정규분포에 가까워진다")

# ── 6. 시각화 ─────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))  # 2행 3열 = 6개 차트 배치, figsize=(가로 15인치, 세로 8인치)

# 정규분포
axes[0, 0].hist(data_normal, bins=30, density=True, alpha=0.7, edgecolor='black')
# ↑ hist: 히스토그램 그리기. bins=막대 30개, density=True→확률밀도로 변환, alpha=투명도 0.7, edgecolor=막대 테두리색
x = np.linspace(140, 200, 100)   # 140~200 사이를 100등분한 x좌표 배열 (이론적 곡선을 그리기 위해)
axes[0, 0].plot(x, stats.norm.pdf(x, mean, std), 'r-', linewidth=2, label='PDF')
# ↑ 정규분포의 확률밀도함수(PDF) 곡선 그리기. 'r-'=빨간 실선, linewidth=선 두께
axes[0, 0].axvline(mean, color='red', linestyle='--', label=f'Mean={mean:.0f}')
# ↑ axvline: 세로 점선 그리기 — 평균 위치 표시. linestyle='--'=점선
axes[0, 0].axvline(mean-std, color='orange', linestyle=':', alpha=0.7)
# ↑ 평균 - 1σ 위치에 주황색 점선
axes[0, 0].axvline(mean+std, color='orange', linestyle=':', alpha=0.7, label=f'±1σ')
# ↑ 평균 + 1σ 위치에 주황색 점선
axes[0, 0].set_title('Normal Distribution')   # 차트 제목
axes[0, 0].legend(fontsize=8)                  # 범례 표시, 글씨 크기 8

# 이항분포
axes[0, 1].hist(data_binom, bins=range(12), density=True, alpha=0.7,
                edgecolor='black', align='left')
# ↑ bins=range(12)→0~11까지 정수 구간, align='left'→막대를 왼쪽 정렬
axes[0, 1].set_title('Binomial Distribution (n=10, p=0.5)')  # 제목: 이항분포 (10회 시행, 성공확률 0.5)
axes[0, 1].set_xlabel('# of Heads')            # x축 라벨: 앞면 횟수

# 균등분포
axes[0, 2].hist(data_uniform, bins=20, density=True, alpha=0.7, edgecolor='black')
# ↑ bins=20→막대 20개로 나눔
axes[0, 2].set_title('Uniform Distribution')    # 제목: 균등분포

# 치우친 분포 (변환 전)
axes[1, 0].hist(data_skewed, bins=30, density=True, alpha=0.7, edgecolor='black')
axes[1, 0].axvline(np.mean(data_skewed), color='red', linestyle='--', label='Mean')
# ↑ 평균 위치에 빨간 점선 — 치우친 분포에서는 평균이 중앙값보다 오른쪽에 있음
axes[1, 0].axvline(np.median(data_skewed), color='blue', linestyle='--', label='Median')
# ↑ 중앙값 위치에 파란 점선
axes[1, 0].set_title('Right Skewed (Before)')   # 제목: 오른쪽으로 치우친 분포 (변환 전)
axes[1, 0].legend(fontsize=8)

# 로그 변환 후
axes[1, 1].hist(data_log, bins=30, density=True, alpha=0.7, edgecolor='black', color='green')
# ↑ color='green'→막대 색상 초록색
axes[1, 1].set_title('After Log Transform')     # 제목: 로그 변환 후 (정규분포에 가까워짐)

# 중심극한정리 시연
sample_means = [np.mean(np.random.exponential(30, 50)) for _ in range(1000)]
# ↑ 지수분포(치우친 분포)에서 50개씩 뽑아 평균을 구하는 것을 1000번 반복
# ↑ 핵심: 어떤 분포든 표본 평균을 모으면 정규분포가 된다 = 중심극한정리
axes[1, 2].hist(sample_means, bins=30, density=True, alpha=0.7, edgecolor='black', color='purple')
# ↑ color='purple'→보라색
axes[1, 2].set_title('Central Limit Theorem\n(sample means of skewed data)')
# ↑ 제목: 중심극한정리 (치우친 데이터의 표본 평균들)

plt.tight_layout()                # 차트 간 간격 자동 조정 — 겹치지 않게
plt.savefig('1-2_output.png', dpi=100)  # 그래프를 이미지 파일로 저장. dpi=해상도 (100 도트/인치)
plt.show()                        # 화면에 그래프 표시

# ── 정리 ──────────────────────────────────
print("\n" + "="*50)              # 구분선 출력 (= 기호 50개)
print("핵심 정리:")
print("  정규분포  → 가장 중요, 68-95-99.7 법칙")
print("  이항분포  → 성공/실패 문제 (이진분류 기초)")
print("  치우침    → 평균 vs 중앙값으로 확인")
print("  로그변환  → 치우친 분포를 정규분포에 가깝게")
print("  중심극한정리 → 표본 평균은 항상 정규분포")
print("="*50)
