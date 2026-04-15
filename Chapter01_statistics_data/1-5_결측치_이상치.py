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

import numpy as np                # numpy: 수학 계산 라이브러리 (배열, 난수, 통계 등)
import pandas as pd               # pandas: 표(테이블) 형태 데이터를 다루는 라이브러리
import matplotlib.pyplot as plt   # matplotlib: 그래프/차트를 그리는 라이브러리

# ── 1. 결측치가 있는 데이터 만들기 ────────
np.random.seed(42)                # 랜덤 시드 고정 — 실행할 때마다 같은 결과가 나오게
n = 100                           # 데이터 100개 생성할 예정
df = pd.DataFrame({               # DataFrame: 표 형태의 데이터 생성
    'age': np.random.randint(20, 60, n).astype(float),  # 20~59세 사이 정수 100개, float로 변환 (NaN 넣으려면 float이어야 함)
    'salary': np.random.normal(50000, 15000, n),         # 평균 5만, 표준편차 1.5만인 정규분포로 연봉 생성
    'department': np.random.choice(['Sales', 'Engineering', 'HR', None], n),  # 4개 중 랜덤 선택 (None=결측치 포함)
})
# 결측치 삽입
df.loc[np.random.choice(n, 10, replace=False), 'age'] = np.nan
# ↑ 100개 행 중 랜덤 10개를 골라 age를 NaN(결측)으로 변경. replace=False→중복 선택 방지
df.loc[np.random.choice(n, 5, replace=False), 'salary'] = np.nan
# ↑ 랜덤 5개 행의 salary를 NaN으로 변경

print("[ 원본 데이터 ]")
print(f"shape: {df.shape}")       # (행 수, 열 수) 출력
print(f"\n결측치 현황:")
print(df.isnull().sum())          # 열별 결측치 개수 출력
print(f"\n처음 10행:")
print(df.head(10))                # 처음 10행 미리보기

# ── 2. 결측치 처리 ────────────────────────
df_cleaned = df.copy()            # copy(): 원본을 건드리지 않고 복사본에서 작업 (원본 보존용)

# 연속형: 중앙값으로 대체 (이상치에 강함)
df_cleaned['age'].fillna(df_cleaned['age'].median(), inplace=True)
# ↑ fillna: NaN을 지정한 값으로 채움. median()=중앙값. inplace=True→원본 데이터 직접 수정
# ↑ 왜 중앙값? 평균은 이상치에 끌려가지만 중앙값은 안정적
df_cleaned['salary'].fillna(df_cleaned['salary'].median(), inplace=True)
# ↑ salary의 NaN도 중앙값으로 대체

# 범주형: 최빈값으로 대체
df_cleaned['department'].fillna(df_cleaned['department'].mode()[0], inplace=True)
# ↑ mode(): 가장 자주 나오는 값(최빈값). [0]은 첫 번째 최빈값 선택

print(f"\n[ 결측치 처리 후 ]")
print(df_cleaned.isnull().sum())  # 처리 후 결측치 수 확인 — 모두 0이어야 정상

# ── 3. 이상치 탐지 (IQR 방법) ─────────────
print(f"\n[ 이상치 탐지 — IQR 방법 ]")

def detect_outliers_iqr(series, name):
    """IQR 방법으로 이상치를 탐지하는 함수"""
    q1 = series.quantile(0.25)    # Q1: 하위 25% 지점의 값 (1사분위수)
    q3 = series.quantile(0.75)    # Q3: 하위 75% 지점의 값 (3사분위수)
    iqr = q3 - q1                 # IQR: 사분위 범위 = Q3 - Q1 (데이터의 중간 50%가 퍼진 정도)
    lower = q1 - 1.5 * iqr       # 하한선: Q1에서 IQR의 1.5배를 뺀 값 — 이 아래는 이상치
    upper = q3 + 1.5 * iqr       # 상한선: Q3에서 IQR의 1.5배를 더한 값 — 이 위는 이상치
    outliers = series[(series < lower) | (series > upper)]
    # ↑ 하한선 미만이거나 상한선 초과인 값을 이상치로 판정
    print(f"  {name}: Q1={q1:.0f}, Q3={q3:.0f}, IQR={iqr:.0f}")
    print(f"    범위: {lower:.0f} ~ {upper:.0f}")
    print(f"    이상치: {len(outliers)}개")
    return lower, upper           # 하한선, 상한선 반환 (나중에 클리핑에 사용)

lower_s, upper_s = detect_outliers_iqr(df_cleaned['salary'], 'salary')
# ↑ salary 열에 대해 IQR 이상치 탐지 실행, 하한/상한 값 저장

# ── 4. 이상치 탐지 (Z-score 방법) ─────────
print(f"\n[ 이상치 탐지 — Z-score 방법 ]")
z_scores = (df_cleaned['salary'] - df_cleaned['salary'].mean()) / df_cleaned['salary'].std()
# ↑ Z-score = (값 - 평균) / 표준편차. 평균에서 몇 σ(표준편차) 떨어져 있는지 계산
outliers_z = df_cleaned[np.abs(z_scores) > 3]
# ↑ |Z| > 3인 데이터 = 평균에서 3σ 이상 떨어진 값 = 이상치로 판정 (99.7% 밖)
print(f"  |Z| > 3인 이상치: {len(outliers_z)}개")

# ── 5. 이상치 처리 (클리핑) ───────────────
df_clipped = df_cleaned.copy()    # 복사본 생성
df_clipped['salary'] = df_clipped['salary'].clip(lower=lower_s, upper=upper_s)
# ↑ clip: 하한선 미만 → 하한선으로, 상한선 초과 → 상한선으로 강제 변환
# ↑ 이상치를 삭제하지 않고, 경계값으로 "깎아내는" 방법

print(f"\n[ 클리핑 처리 후 ]")
print(f"  salary 범위: {df_cleaned['salary'].min():.0f}~{df_cleaned['salary'].max():.0f}")
print(f"  → 클리핑 후: {df_clipped['salary'].min():.0f}~{df_clipped['salary'].max():.0f}")

# ── 6. 시각화 ─────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))  # 2행 3열 = 6개 차트, figsize=(가로 15인치, 세로 8인치)

# 결측치 시각화
missing = df.isnull().sum()       # 원본 데이터의 열별 결측치 수
axes[0, 0].bar(missing.index, missing.values, color=['red' if v > 0 else 'green' for v in missing.values])
# ↑ bar: 막대 차트. 결측치가 있으면 빨간색, 없으면 초록색으로 표시
axes[0, 0].set_title('Missing Values per Column')    # 제목: 열별 결측치 수
axes[0, 0].tick_params(axis='x', rotation=45)        # x축 라벨을 45도 기울여서 표시 (겹침 방지)

# 처리 전 salary 분포
axes[0, 1].hist(df['salary'].dropna(), bins=20, edgecolor='black', alpha=0.7)
# ↑ dropna(): NaN을 제거한 후 히스토그램. bins=막대 20개
axes[0, 1].set_title('Salary (Before)')              # 제목: 결측치 처리 전 salary 분포

# 처리 후 salary 분포
axes[0, 2].hist(df_cleaned['salary'], bins=20, edgecolor='black', alpha=0.7, color='green')
# ↑ 결측치를 중앙값으로 채운 후의 분포. color='green'→초록색
axes[0, 2].set_title('Salary (After fillna)')        # 제목: 결측치 처리 후

# IQR 이상치 시각화
axes[1, 0].boxplot(df_cleaned['salary'])             # boxplot: 상자그림으로 이상치(동그라미 점) 확인
axes[1, 0].set_title('Salary Box Plot (IQR)')        # 제목: IQR 기반 이상치 확인

# Z-score 분포
axes[1, 1].hist(z_scores, bins=20, edgecolor='black', alpha=0.7, color='purple')
# ↑ Z-score 분포를 히스토그램으로. 대부분 -3~+3 사이에 있어야 정상
axes[1, 1].axvline(-3, color='red', linestyle='--', label='Z=-3')
# ↑ Z=-3 위치에 빨간 점선 — 이 왼쪽은 이상치
axes[1, 1].axvline(3, color='red', linestyle='--', label='Z=+3')
# ↑ Z=+3 위치에 빨간 점선 — 이 오른쪽은 이상치
axes[1, 1].set_title('Z-score Distribution')         # 제목: Z-score 분포
axes[1, 1].legend()                                   # 범례 표시 (Z=-3, Z=+3)

# 클리핑 전후 비교
axes[1, 2].boxplot([df_cleaned['salary'], df_clipped['salary']],
                    tick_labels=['Before Clip', 'After Clip'])
# ↑ 클리핑 전후 salary를 상자그림으로 나란히 비교. tick_labels=각 상자의 라벨
axes[1, 2].set_title('Before vs After Clipping')     # 제목: 클리핑 전 vs 후

plt.tight_layout()               # 차트 간 간격 자동 조정
plt.savefig('1-4_output.png', dpi=100)  # 이미지 파일로 저장. dpi=해상도
plt.show()                       # 화면에 그래프 표시

# ── 정리 ──────────────────────────────────
print("\n" + "="*50)             # 구분선 출력
print("핵심 정리:")
print("  결측치 → isnull().sum()으로 확인")
print("  연속형 결측 → 중앙값(median) 대체")
print("  범주형 결측 → 최빈값(mode) 대체")
print("  이상치 탐지 → IQR 또는 Z-score")
print("  이상치 처리 → 제거, 클리핑, 또는 유지")
print("="*50)
