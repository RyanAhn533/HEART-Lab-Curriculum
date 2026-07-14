# ============================================
# 2-1. 상관분석 (Correlation Analysis)
#
# 왜 배우는가:
#   "이 특성(X)이 결과(Y)와 관련이 있는가?"
#   관련 없는 특성을 넣으면 모델 성능이 떨어진다.
#
# 나중에 만나는 곳:
#   → 2-2 선형회귀: 상관 높은 특성으로 예측
#   → 후속 챕터 Feature Importance: 특성 선택
#
# ▶ 보고 오기: StatQuest "Correlation"
#
# Ref: Stanford CS229 W1 / Google MLCC
# ============================================

import numpy as np                                  # numpy: 수학 계산 라이브러리 (배열, 평균, 표준편차 등)
import pandas as pd                                  # pandas: 표(테이블) 형태 데이터를 다루는 라이브러리
import matplotlib.pyplot as plt                      # matplotlib: 그래프를 그리는 라이브러리
import seaborn as sns                                # seaborn: matplotlib 위에 만든 예쁜 시각화 라이브러리
from sklearn.datasets import fetch_california_housing  # fetch_california_housing: 캘리포니아 집값 예제 데이터셋을 가져오는 함수

# ── 1. 데이터 로드 ────────────────────────
data = fetch_california_housing()                    # 캘리포니아 집값 데이터를 다운로드해서 data에 저장
df = pd.DataFrame(data.data, columns=data.feature_names)  # 숫자 데이터를 표(DataFrame)로 변환, 열 이름을 특성 이름으로 지정
df['Price'] = data.target                            # 집값(target)을 'Price'라는 새 열로 추가

# ── 2. 상관계수 계산 ──────────────────────
corr_matrix = df.corr()                              # corr(): 모든 열 쌍의 상관계수를 계산한 표를 만듦 (-1~+1 사이 값)
corr_with_price = corr_matrix['Price'].drop('Price').sort_values(ascending=False)  # Price와의 상관계수만 뽑고, 자기 자신은 빼고, 큰 순서로 정렬

print("[ Price와의 상관계수 (Pearson) ]")
for col, val in corr_with_price.items():             # 각 특성과 Price의 상관계수를 하나씩 꺼냄
    strength = "강함" if abs(val) > 0.7 else "중간" if abs(val) > 0.4 else "약함" if abs(val) > 0.2 else "거의 없음"
    # ↑ 강도 판정 기준: |r|>0.7 강함, |r|>0.4 중간, |r|>0.2 약함, 그 이하 거의 없음
    print(f"  {col:12s}: {val:+.3f}  ({strength})")  # 특성 이름, 상관계수, 강도를 출력

# ── 3. 상관 ≠ 인과 예시 ──────────────────
print("\n[ 주의: 상관 ≠ 인과 ]")                     # 상관관계가 있다고 원인-결과는 아니라는 경고
print("  Latitude와 Price 상관 = -0.14")
print("  → 위도가 가격을 결정하는 게 아니라,")
print("  → 특정 위도(LA, SF)에 비싼 집이 많은 것")

# ── 4. Spearman & 상관 0 ≠ 무관계 ────────
print("\n[ Spearman — 순위 기반 상관 ]")
spearman_price = df.corr(method='spearman')['Price'].drop('Price')  # method='spearman': 값 대신 순위(rank)로 계산 — 곡선·이상치에 강함
print(f"  MedInc: Pearson={corr_with_price['MedInc']:+.3f}, Spearman={spearman_price['MedInc']:+.3f}")

print("\n[ 상관 0 ≠ 관계 없음 ]")
x_curve = np.linspace(-3, 3, 100)                     # -3~3 사이 100개 점
y_curve = x_curve ** 2                                 # y = x²: 완벽한 관계 (포물선)
r_curve = np.corrcoef(x_curve, y_curve)[0, 1]          # Pearson 상관계수 계산
print(f"  y = x² 인데 r = {r_curve:.3f}")
print(f"  → 직선 관계가 아니면 Pearson이 못 잡는다. scatter를 꼭 그려볼 것")

# ── 5. 시각화 ─────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))      # 2행 3열 = 6개 그래프 영역 생성, 전체 크기 15x9인치

# 상관 heatmap
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',  # heatmap: 상관계수를 색깔로 표현한 표
            center=0, ax=axes[0, 0], annot_kws={'size': 7})       # annot=True: 숫자 표시, fmt='.2f': 소수점 2자리, cmap: 색상표, center=0: 0을 중심으로 색 배치
axes[0, 0].set_title('Correlation Heatmap')          # 그래프 제목 설정

# 상관계수 바 차트
colors = ['steelblue' if v > 0 else 'salmon' for v in corr_with_price.values]  # 양의 상관 = 파란색, 음의 상관 = 빨간색
axes[0, 1].barh(corr_with_price.index, corr_with_price.values, color=colors)   # barh: 가로 막대 그래프 (특성별 상관계수)
axes[0, 1].set_title('Correlation with Price')       # 그래프 제목 설정
axes[0, 1].axvline(0, color='black', linewidth=0.5)  # axvline: x=0 위치에 세로 기준선 추가

# 강한 양의 상관: MedInc vs Price
axes[0, 2].scatter(df['MedInc'], df['Price'], alpha=0.2, s=3)  # scatter: 산점도 (alpha=0.2: 투명도, s=3: 점 크기)
z = np.polyfit(df['MedInc'], df['Price'], 1)         # polyfit: 1차 다항식(직선) 피팅, 기울기와 절편을 구함
p = np.poly1d(z)                                     # poly1d: 피팅 결과로 다항식 함수 객체 생성
x_line = np.linspace(df['MedInc'].min(), df['MedInc'].max(), 100)  # linspace: 최솟값~최댓값 사이를 100등분한 점 생성
axes[0, 2].plot(x_line, p(x_line), 'r-', linewidth=2, label=f'r={corr_matrix.loc["MedInc","Price"]:.3f}')  # 추세선(빨간 직선) 그리기, 범례에 상관계수 표시
axes[0, 2].set_xlabel('MedInc')                      # x축 라벨: 중위 소득
axes[0, 2].set_ylabel('Price')                        # y축 라벨: 집값
axes[0, 2].set_title('Strong Positive Correlation')   # 제목: 강한 양의 상관
axes[0, 2].legend()                                   # legend: 범례 표시

# 약한 상관: Population vs Price
axes[1, 0].scatter(df['Population'], df['Price'], alpha=0.2, s=3)  # 인구수 vs 집값 산점도
axes[1, 0].set_xlabel('Population')                   # x축 라벨: 인구수
axes[1, 0].set_ylabel('Price')                        # y축 라벨: 집값
axes[1, 0].set_title(f'Weak Correlation (r={corr_matrix.loc["Population","Price"]:.3f})')  # 제목에 상관계수 표시

# 음의 상관: Latitude vs Price
axes[1, 1].scatter(df['Latitude'], df['Price'], alpha=0.2, s=3)   # 위도 vs 집값 산점도
axes[1, 1].set_xlabel('Latitude')                     # x축 라벨: 위도
axes[1, 1].set_ylabel('Price')                        # y축 라벨: 집값
axes[1, 1].set_title(f'Negative Correlation (r={corr_matrix.loc["Latitude","Price"]:.3f})')  # 제목에 음의 상관계수 표시

# 상관 없음 예시: 랜덤
np.random.seed(42)                                    # 난수 시드 고정: 실행할 때마다 같은 결과가 나오게 함
x_rand = np.random.randn(500)                         # randn: 표준정규분포에서 500개 랜덤 값 생성 (x축)
y_rand = np.random.randn(500)                         # randn: 표준정규분포에서 500개 랜덤 값 생성 (y축, x와 무관)
axes[1, 2].scatter(x_rand, y_rand, alpha=0.3, s=10)  # 무관한 두 변수의 산점도 (패턴 없음)
r_rand = np.corrcoef(x_rand, y_rand)[0, 1]           # corrcoef: 두 변수 간 상관계수 계산 (0에 가까울 것)
axes[1, 2].set_title(f'No Correlation (r={r_rand:.3f})')  # 제목에 거의 0인 상관계수 표시

plt.tight_layout()                                    # tight_layout: 그래프 간 간격을 자동 조정해서 겹침 방지
plt.savefig('2-1_output.png', dpi=100)               # savefig: 그래프를 PNG 파일로 저장 (dpi=100: 해상도)
plt.show()                                            # show: 화면에 그래프 표시

# ── 정리 ──────────────────────────────────
print("\n" + "="*50)
print("핵심 정리:")
print("  상관계수 = -1 ~ +1 (관계의 방향과 강도)")
print("  |r| > 0.7 강함, |r| > 0.4 중간 → 쓸만한 특성")
print("  상관 ≠ 인과 (항상 주의)")
print("  상관 0 ≠ 무관계 (y=x²) → scatter 확인")
print("  scatter + 추세선 = 다음에 배울 선형회귀")
print("="*50)
