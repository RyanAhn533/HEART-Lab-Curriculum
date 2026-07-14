# ============================================
# 1-1. 기술통계 (Descriptive Statistics)
#
# 왜 배우는가:
#   데이터를 모델에 넣기 전에 "어떻게 생겼는지" 알아야 한다.
#   기술통계는 데이터를 몇 개의 숫자로 요약하는 방법이다.
#
# 나중에 만나는 곳:
#   → 1-6 스케일링: StandardScaler = (x - mean) / std
#   → 1-5 이상치 처리: IQR 기준 제거
#   → 1-4 EDA: pandas describe()
#
# ▶ 보고 오기: 구글 "기술통계 파이썬"
#
# Ref: Stanford CS229 W1 / Google MLCC "Numerical Data"
# ============================================

import numpy as np                # numpy: 수학 계산 라이브러리 (배열, 평균, 표준편차 등)
import matplotlib.pyplot as plt   # matplotlib: 그래프 그리기 라이브러리

# ── 1. 데이터 준비 ────────────────────────

np.random.seed(42)                # 랜덤 시드 고정 — 실행할 때마다 같은 결과가 나오게
# np.random.normal: 정규분포를 따르는 난수 생성
#   loc=70: 평균 70점, scale=12: 표준편차 12, size=30: 30명
scores = np.random.normal(loc=70, scale=12, size=30).astype(int)  # .astype(int): 소수점 버림
scores = np.append(scores, [15, 98])  # 이상치 2개 일부러 추가 (극단적으로 낮은/높은 점수)

print(f"데이터: {sorted(scores)}")     # sorted(): 오름차순 정렬해서 보기
print(f"데이터 개수: {len(scores)}")    # len(): 데이터 개수

# ── 2. 중심 경향 ──────────────────────────

mean = np.mean(scores)       # np.mean(): 평균 — 전부 더해서 개수로 나눈 것
median = np.median(scores)   # np.median(): 중앙값 — 정렬했을 때 가운데 값

print(f"\n[ 중심 경향 ]")
print(f"평균(Mean):   {mean:.1f}")     # :.1f = 소수점 1자리까지 출력
print(f"중앙값(Median): {median:.1f}")
print(f"→ 평균과 중앙값이 다르면 이상치가 있을 수 있다")

# ── 3. 산포도 ─────────────────────────────

var = np.var(scores)   # np.var(): 분산 — 각 값이 평균에서 떨어진 거리²의 평균
std = np.std(scores)   # np.std(): 표준편차 — 분산에 루트(√) 씌운 것, 원래 단위로 복원

print(f"\n[ 산포도 ]")
print(f"분산(Variance):       {var:.1f}")
print(f"표준편차(Std Dev):     {std:.1f}")
print(f"최소~최대: {np.min(scores)} ~ {np.max(scores)}")   # min/max: 최소값/최대값
print(f"→ 데이터가 평균에서 약 {std:.0f}점 정도 퍼져있다")

# ── 4. 사분위수 / IQR ────────────────────

q1 = np.percentile(scores, 25)   # Q1: 하위 25% 지점의 값
q3 = np.percentile(scores, 75)   # Q3: 상위 25% 지점의 값
iqr = q3 - q1                    # IQR: Q3 - Q1 (데이터 중간 50%의 범위)
lower = q1 - 1.5 * iqr           # 이상치 하한: 이보다 작으면 이상치
upper = q3 + 1.5 * iqr           # 이상치 상한: 이보다 크면 이상치

print(f"\n[ 사분위수 ]")
print(f"Q1 (25%): {q1:.1f}")
print(f"Q2 (50%): {median:.1f}")
print(f"Q3 (75%): {q3:.1f}")
print(f"IQR:      {iqr:.1f}")
print(f"이상치 기준: < {lower:.1f} 또는 > {upper:.1f}")

# 이상치 찾기: lower 미만 또는 upper 초과인 값들
outliers = scores[(scores < lower) | (scores > upper)]   # | = "또는" 조건
print(f"이상치: {outliers}")

# ── 5. 시각화 ─────────────────────────────

# subplots: 그래프 여러 개를 한 줄에 배치 (1행 3열)
fig, axes = plt.subplots(1, 3, figsize=(14, 4))   # figsize: 전체 그림 크기 (가로, 세로)

# 히스토그램: 데이터 분포를 막대 그래프로 보여줌
axes[0].hist(scores, bins=10, edgecolor='black', alpha=0.7)   # bins=10: 10개 구간으로 나눔
axes[0].axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean={mean:.1f}')     # 평균 선
axes[0].axvline(median, color='blue', linestyle='--', linewidth=2, label=f'Median={median:.1f}')  # 중앙값 선
axes[0].set_title('Score Distribution (Histogram)')   # 그래프 제목
axes[0].set_xlabel('Score')                            # X축 이름
axes[0].set_ylabel('Frequency')                        # Y축 이름
axes[0].legend()                                       # 범례 표시

# 박스플롯: 사분위수와 이상치를 한눈에 보여줌
bp = axes[1].boxplot(scores, vert=True, patch_artist=True)   # patch_artist: 색 채우기
bp['boxes'][0].set_facecolor('lightblue')                     # 박스 색상
axes[1].set_title('Score Distribution (Box Plot)')
axes[1].set_ylabel('Score')

# 이상치 scatter: 정상 데이터와 이상치를 다른 색으로 표시
normal = scores[(scores >= lower) & (scores <= upper)]   # & = "그리고" 조건
axes[2].scatter(range(len(normal)), sorted(normal),       # scatter: 점 그래프
                color='blue', alpha=0.6, label='Normal')  # alpha: 투명도 (0~1)
axes[2].scatter(range(len(normal), len(normal)+len(outliers)), sorted(outliers),
                color='red', s=100, marker='x', linewidths=2, label='Outlier')  # 이상치는 빨간 X
axes[2].axhline(lower, color='red', linestyle=':', alpha=0.5, label=f'Lower={lower:.0f}')  # 하한선
axes[2].axhline(upper, color='red', linestyle=':', alpha=0.5, label=f'Upper={upper:.0f}')  # 상한선
axes[2].set_title('Normal vs Outlier')
axes[2].set_ylabel('Score')
axes[2].legend(fontsize=8)   # fontsize: 범례 글자 크기

plt.tight_layout()              # 그래프 간격 자동 조정
plt.savefig('1-1_output.png', dpi=100)   # 그래프를 이미지 파일로 저장
plt.show()                      # 화면에 그래프 표시

# ── 6. pandas로 한 번에 ──────────────────

import pandas as pd              # pandas: 데이터 분석 라이브러리 (표 형태로 데이터 다루기)
df = pd.DataFrame({'score': scores})   # DataFrame: 엑셀 같은 표 형태로 변환

print("\n[ pandas describe() — 실무에서는 이걸 쓴다 ]")
print(df.describe())   # describe(): 평균, 표준편차, 사분위수 등을 한 번에 출력

# ddof 비교 — np.std 기본은 n으로 나눔(ddof=0), describe()는 n-1로 나눔(ddof=1)
print(f"\nnp.std (ddof=0, n으로 나눔):  {np.std(scores):.4f}")
print(f"np.std (ddof=1, n-1로 나눔): {np.std(scores, ddof=1):.4f}  ← describe()의 std와 같은 값")

# ── 정리 ──────────────────────────────────
print("\n" + "="*50)
print("핵심 정리:")
print("  평균/중앙값 → 데이터의 중심")
print("  표준편차    → 데이터의 퍼짐 정도")
print("  IQR        → 이상치 판별 기준")
print("  describe()  → 실무에서 가장 먼저 치는 명령어")
print("="*50)
