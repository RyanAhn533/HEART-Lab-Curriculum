# ============================================
# 1-4. 데이터 탐색 (EDA: Exploratory Data Analysis)
#
# 왜 배우는가:
#   모델에 넣기 전에 "눈으로 먼저 보는 것".
#   실무에서 가장 많은 시간을 쓰는 단계.
#
# 나중에 만나는 곳:
#   → 모든 Chapter의 #1: 데이터 로드 후 가장 먼저
#   → 1-6 인코딩: 범주형 데이터 변환
#   → 2-1 상관분석: heatmap
#
# ▶ 보고 오기: 구글 "pandas EDA 실습"
#
# Ref: Google MLCC "Numerical/Categorical Data"
# ============================================

import numpy as np                # numpy: 수학 계산 라이브러리 (배열, 평균, 표준편차 등)
import pandas as pd               # pandas: 표(테이블) 형태의 데이터를 다루는 라이브러리
import matplotlib.pyplot as plt   # matplotlib: 그래프/차트를 그리는 라이브러리
import seaborn as sns             # seaborn: matplotlib 기반 통계 시각화 라이브러리 (히트맵 등 예쁜 차트)

# ── 1. 데이터 로드 ────────────────────────
# sklearn 내장 데이터셋 (캘리포니아 집값)
from sklearn.datasets import fetch_california_housing  # sklearn: 머신러닝 라이브러리. 연습용 데이터셋 제공
data = fetch_california_housing()                      # 캘리포니아 주택 가격 데이터 불러오기 (20,640건)
df = pd.DataFrame(data.data, columns=data.feature_names)  # data.data=숫자 배열을 표(DataFrame)로 변환, 열 이름 지정
df['Price'] = data.target         # data.target=예측할 값(집값)을 'Price' 열로 추가

# ── 2. 기본 탐색 (이 6줄을 습관으로) ──────
print("[ 1) shape ]")
print(f"  {df.shape[0]}행, {df.shape[1]}열\n")       # shape: (행 수, 열 수) — 데이터 크기 확인

print("[ 2) columns & dtypes ]")
print(df.dtypes)                  # dtypes: 각 열의 데이터 타입 (float64=실수, int64=정수, object=문자열)

print(f"\n[ 3) head ]")
print(df.head())                  # head(): 처음 5행만 미리보기 — 데이터가 어떻게 생겼는지 확인

print(f"\n[ 4) describe ]")
print(df.describe())             # describe(): 각 열의 통계 요약 (평균, 표준편차, 최솟값, 사분위수, 최댓값)

print(f"\n[ 5) 결측치 ]")
print(df.isnull().sum())         # isnull(): 각 셀이 비어있는지 True/False, sum(): True 개수 합산 → 열별 결측치 수
print(f"  → 결측치 총 {df.isnull().sum().sum()}개")  # .sum().sum(): 전체 결측치 총합

print(f"\n[ 6) info ]")
print(df.info())                 # info(): 열 이름, 타입, 결측치 수, 메모리 사용량 한눈에 보기

# ── 3. 시각화 ─────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))  # 2행 3열 = 6개 차트 배치, figsize=(가로 15인치, 세로 9인치)

# 타겟 변수 분포
axes[0, 0].hist(df['Price'], bins=30, edgecolor='black', alpha=0.7)
# ↑ 집값(Price) 히스토그램. bins=막대 30개, edgecolor=테두리색, alpha=투명도
axes[0, 0].set_title('Target: Price Distribution')  # 제목: 타겟 변수(집값) 분포
axes[0, 0].set_xlabel('Price ($100k)')               # x축 라벨: 가격 (단위: 10만 달러)

# 주요 특성 분포
axes[0, 1].hist(df['MedInc'], bins=30, edgecolor='black', alpha=0.7, color='green')
# ↑ MedInc(중위 소득) 분포 히스토그램. color='green'→초록색 막대
axes[0, 1].set_title('MedInc (Median Income)')       # 제목: 중위 소득

# 박스플롯 (이상치 확인)
axes[0, 2].boxplot([df['MedInc'], df['HouseAge'], df['AveRooms']],
                    tick_labels=['MedInc', 'HouseAge', 'AveRooms'])
# ↑ boxplot: 상자그림 — 중앙값, 사분위수, 이상치(동그라미 점)를 한눈에 보여줌
# ↑ 3개 변수를 나란히 비교. tick_labels=각 상자의 이름표 (matplotlib 3.9+에서 labels→tick_labels로 변경)
axes[0, 2].set_title('Box Plot (Outlier Check)')     # 제목: 이상치 확인용 상자그림

# Scatter (두 변수 관계)
axes[1, 0].scatter(df['MedInc'], df['Price'], alpha=0.3, s=5)
# ↑ scatter: 산점도 — 소득(x)과 집값(y)의 관계를 점으로 표시
# ↑ alpha=0.3→점 투명도 (겹치는 영역이 보이게), s=5→점 크기
axes[1, 0].set_xlabel('MedInc')                      # x축: 중위 소득
axes[1, 0].set_ylabel('Price')                        # y축: 집값
axes[1, 0].set_title('MedInc vs Price')              # 제목: 소득 vs 집값 관계

# 상관관계 Heatmap
corr = df.corr()                 # corr(): 모든 열 쌍의 상관계수 계산 (-1~+1). 1에 가까우면 양의 상관, -1에 가까우면 음의 상관 (자세한 건 2-1에서)
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=axes[1, 1], annot_kws={'size': 7})
# ↑ heatmap: 상관계수를 색깔로 시각화. annot=True→숫자 표시, fmt='.2f'→소수점 2자리
# ↑ cmap='coolwarm'→파란색(음의 상관)~빨간색(양의 상관) 색상표
# ↑ center=0→0을 색상 중심으로, ax=어떤 차트에 그릴지, annot_kws→숫자 글씨 크기
axes[1, 1].set_title('Correlation Heatmap')          # 제목: 상관관계 히트맵

# 상위 상관 특성
corr_price = corr['Price'].drop('Price').sort_values(ascending=False)
# ↑ Price와 다른 모든 열의 상관계수를 구하고, Price 자기 자신 제외 후, 높은 순으로 정렬
axes[1, 2].barh(corr_price.index, corr_price.values, color='steelblue')
# ↑ barh: 수평 막대 차트. index=변수 이름, values=상관계수 값
axes[1, 2].set_title('Correlation with Price')       # 제목: 집값과의 상관관계
axes[1, 2].set_xlabel('Correlation')                  # x축: 상관계수

plt.tight_layout()               # 차트 간 간격 자동 조정 — 겹치지 않게
plt.savefig('1-4_output.png', dpi=100)  # 그래프를 이미지 파일로 저장. dpi=해상도
plt.show()                       # 화면에 그래프 표시

# ── 4. EDA 체크리스트 결과 ────────────────
print("\n" + "="*50)             # 구분선 출력
print("EDA 체크리스트 결과:")
print(f"  shape: {df.shape}")                         # 데이터 크기
print(f"  결측치: {df.isnull().sum().sum()}개")         # 전체 결측치 수
print(f"  타겟(Price) 평균: {df['Price'].mean():.2f}") # 집값 평균
print(f"  타겟(Price) 치우침: {'우로 치우침' if df['Price'].skew() > 0 else '좌로 치우침'}")
# ↑ skew(): 왜도 계산. 양수면 오른쪽으로 치우침, 음수면 왼쪽으로 치우침
print(f"  Price와 가장 상관 높은 특성: {corr_price.index[0]} ({corr_price.values[0]:.3f})")
# ↑ 집값과 가장 관련이 깊은 변수 출력
print("="*50)
