# ============================================
# 1-3. 데이터를 다루는 도구 — numpy, pandas, matplotlib
#
# 왜 배우는가:
#   1-1, 1-2에서 np.mean(), plt.hist()를 썼는데
#   이게 정확히 뭔지 정리 + pandas 추가.
#   이후 모든 코드에서 이 세 가지를 계속 쓴다.
#
# 나중에 만나는 곳:
#   → 모든 Chapter: numpy 배열, pandas DataFrame, matplotlib 그래프
#   → 1-6 인코딩: 원핫 인코딩
#   → 3-4 KNN: 벡터 거리
#   → Chapter 7: 임베딩 (벡터 개념 확장)
#
# ▶ 보고 오기: 구글 "numpy 기초", "pandas 기초"
# ============================================

import numpy as np                # numpy: 배열(숫자 묶음) 연산 라이브러리 — 모든 ML 데이터의 기본 형태
import pandas as pd               # pandas: 표(DataFrame) 형태로 데이터를 다루는 라이브러리 — 엑셀 같은 것
import matplotlib.pyplot as plt   # matplotlib: 그래프 그리기 라이브러리

# ══════════════════════════════════════════
# 1. numpy — 배열(숫자 묶음) 다루기
# ══════════════════════════════════════════

print("[ 1. numpy 기초 ]")

# ── 리스트 vs numpy 배열 ──────────────────
py_list = [1, 2, 3, 4, 5]              # 파이썬 기본 리스트
np_array = np.array([1, 2, 3, 4, 5])   # numpy 배열로 변환

print(f"리스트 + 리스트:  {py_list + py_list}")        # [1,2,3,4,5,1,2,3,4,5] — 이어붙이기
print(f"배열 + 배열:      {np_array + np_array}")       # [2,4,6,8,10] — 원소별 덧셈!
print(f"배열 * 3:         {np_array * 3}")              # [3,6,9,12,15] — 원소별 곱셈
print(f"→ numpy는 수학 연산이 자연스럽다")

# ── shape (크기) ──────────────────────────
a = np.array([1, 2, 3])                         # 1차원 배열
b = np.array([[1, 2, 3], [4, 5, 6]])            # 2차원 배열 (2행 3열)
c = np.zeros((60000, 28, 28))                    # 3차원 배열 (이미지 6만장 × 28 × 28)

print(f"\n1차원 shape: {a.shape}")                # (3,) — 숫자 3개
print(f"2차원 shape: {b.shape}")                  # (2, 3) — 2행 3열
print(f"3차원 shape: {c.shape}")                  # (60000, 28, 28) — 이미지

# ── 인덱싱 / 슬라이싱 ────────────────────
data = np.array([10, 20, 30, 40, 50])
print(f"\n인덱싱: data[0]={data[0]}, data[-1]={data[-1]}")   # [0]: 첫번째, [-1]: 마지막
print(f"슬라이싱: data[1:4]={data[1:4]}")                     # [1:4]: 인덱스 1~3 (4 미포함)

# ── reshape (모양 바꾸기) ────────────────
flat = np.arange(12)                             # [0,1,2,...,11] — 1차원 12개
matrix = flat.reshape(3, 4)                      # 3행 4열로 재배치 (총 개수는 같아야 함)
print(f"\nreshape 전: {flat.shape}")               # (12,)
print(f"reshape 후: {matrix.shape}")               # (3, 4)
print(f"→ 이미지 처리(Chapter 5)에서 reshape 많이 씀")

# ── 행 = 샘플, 열 = 특성 ────────────────
print(f"\n[ 행 = 샘플, 열 = 특성 ]")
from sklearn.datasets import load_iris            # sklearn: ML 라이브러리 — 내장 예제 데이터 제공
iris = load_iris()                                # Iris 데이터 로드 (150송이 붓꽃)
X = iris.data                                     # X: 입력 데이터 (150, 4)
print(f"shape: {X.shape}")                         # (150, 4) = "150개 샘플, 4개 특성"
print(f"  행(150) = 붓꽃 150송이 (샘플)")
print(f"  열(4)   = 꽃받침 길이, 너비, 꽃잎 길이, 너비 (특성)")
print(f"  X[0]    = {X[0]} ← 첫 번째 붓꽃의 4가지 측정값")

# ══════════════════════════════════════════
# 2. 데이터 타입
# ══════════════════════════════════════════

print(f"\n{'='*50}")
print("[ 2. 데이터 타입 ]")

print(f"정수 (int):   {type(3)} → 나이, 개수")
print(f"실수 (float): {type(3.14)} → 키, 온도, 확률")
print(f"문자 (str):   {type('남')} → 범주형 (모델이 못 읽음!)")
print(f"불린 (bool):  {type(True)} → 조건 판단")

arr_int = np.array([1, 2, 3])
arr_float = arr_int.astype(float)                # astype: 타입 변환 (정수 → 실수)
print(f"\nint 배열: {arr_int} (dtype={arr_int.dtype})")
print(f"float 변환: {arr_float} (dtype={arr_float.dtype})")
print(f"→ 모델 입력은 보통 float32")

# ══════════════════════════════════════════
# 3. pandas — 표(DataFrame) 다루기
# ══════════════════════════════════════════

print(f"\n{'='*50}")
print("[ 3. pandas 기초 ]")

# DataFrame 만들기
df = pd.DataFrame({
    'name':   ['Kim', 'Lee', 'Park', 'Choi', 'Jung'],   # 문자열 컬럼
    'age':    [25, 30, 28, 35, 22],                       # 정수 컬럼
    'score':  [85.5, 92.0, 78.3, 88.7, 95.1],            # 실수 컬럼
    'passed': [True, True, False, True, True],            # 불린 컬럼
})

print(f"\n[ DataFrame ]")
print(df)                                                  # 표 형태로 출력
print(f"\nshape: {df.shape}")                               # (5, 4) = 5행 4열
print(f"columns: {list(df.columns)}")                       # 컬럼 이름 목록
print(f"dtypes:\n{df.dtypes}")                              # 각 컬럼의 데이터 타입

# 컬럼 접근
print(f"\ndf['age']:\n{df['age'].values}")                  # 특정 컬럼 가져오기 (.values: numpy 배열로)

# 조건 필터링
print(f"\nage > 25인 사람:")
print(df[df['age'] > 25])                                   # df[조건]: 조건에 맞는 행만

# describe
print(f"\ndescribe():")
print(df.describe())                                        # 기술통계 한 번에 (count, mean, std, min, max...)

# CSV 읽기/쓰기
df.to_csv('sample.csv', index=False)                        # to_csv: DataFrame을 CSV 파일로 저장
df_loaded = pd.read_csv('sample.csv')                       # read_csv: CSV 파일을 DataFrame으로 읽기
print(f"\nCSV 저장 후 읽기: {df_loaded.shape}")
import os; os.remove('sample.csv')                          # 샘플 파일 삭제

# ══════════════════════════════════════════
# 4. matplotlib — 그래프 그리기
# ══════════════════════════════════════════

print(f"\n{'='*50}")
print("[ 4. matplotlib 기초 ]")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))            # 2x2 격자로 4개 그래프 배치

# 선 그래프 (plot)
x = np.linspace(0, 10, 50)                                  # linspace: 0~10을 50등분
axes[0, 0].plot(x, np.sin(x), 'b-', label='sin(x)')        # 'b-': 파란색 실선
axes[0, 0].plot(x, np.cos(x), 'r--', label='cos(x)')       # 'r--': 빨간색 점선
axes[0, 0].set_title('Line Plot (plot)')                     # 그래프 제목
axes[0, 0].set_xlabel('x')                                   # x축 이름
axes[0, 0].set_ylabel('y')                                   # y축 이름
axes[0, 0].legend()                                          # 범례 표시

# 점 그래프 (scatter)
np.random.seed(42)
axes[0, 1].scatter(np.random.randn(50), np.random.randn(50),   # scatter: 점 그래프
                   c=np.random.randn(50), cmap='viridis',       # c: 색상값, cmap: 색상표
                   s=50, alpha=0.7)                              # s: 점 크기, alpha: 투명도
axes[0, 1].set_title('Scatter Plot (scatter)')

# 히스토그램 (hist)
data = np.random.normal(70, 10, 200)                        # 평균 70, 표준편차 10인 데이터 200개
axes[1, 0].hist(data, bins=15, edgecolor='black', alpha=0.7) # bins: 막대 개수, edgecolor: 테두리
axes[1, 0].set_title('Histogram (hist)')

# 막대 그래프 (bar)
categories = ['A', 'B', 'C', 'D']
values = [23, 45, 12, 67]
axes[1, 1].bar(categories, values, color='steelblue',        # bar: 막대 그래프
               edgecolor='black')                             # edgecolor: 테두리 색
axes[1, 1].set_title('Bar Chart (bar)')

plt.tight_layout()                                            # 그래프 간격 자동 조정
plt.savefig('1-3_output.png', dpi=100)                       # 이미지 파일로 저장
plt.show()                                                    # 화면에 표시

# ══════════════════════════════════════════
# 5. 데이터 표현 — 숫자, 범주, 원핫, 벡터
# ══════════════════════════════════════════

print(f"\n{'='*50}")
print("[ 5. 데이터 표현 ]")

# 숫자형 vs 범주형
print("\n숫자형: 키=170.5, 체중=65.0 → 그대로 사용 가능")
print("범주형: 성별='남', 혈액형='A' → 숫자로 변환 필요!")

# 라벨 인코딩
print(f"\n[ 라벨 인코딩 ]")
labels = ['남', '여', '남', '여', '남']
label_map = {'남': 0, '여': 1}                                # 수동 매핑
encoded = [label_map[l] for l in labels]                      # 각 값을 숫자로 변환
print(f"  원본: {labels}")
print(f"  변환: {encoded}")
print(f"  → 간단하지만 '순서'가 생김 (0 < 1)")

# 원핫 인코딩
print(f"\n[ 원핫 인코딩 ]")
blood = ['A', 'B', 'O', 'AB']
for b in blood:
    onehot = [1 if b == t else 0 for t in blood]              # 해당 범주만 1, 나머지 0
    print(f"  {b:2s} → {onehot}")
print(f"  → 순서 없음, 각 범주가 독립적")
print(f"  → Chapter 4 다중분류에서 to_categorical()로 자동 변환")

# 벡터 개념
print(f"\n[ 벡터 = 데이터 1개 = 공간의 한 점 ]")
patient1 = np.array([170, 65, 120, 36.5])                    # 환자1: [키, 체중, 혈압, 체온]
patient2 = np.array([165, 55, 110, 36.8])                    # 환자2
distance = np.sqrt(np.sum((patient1 - patient2)**2))          # 유클리드 거리 (두 점 사이 거리)
print(f"  환자1: {patient1}")
print(f"  환자2: {patient2}")
print(f"  거리:  {distance:.2f}")
print(f"  → '비슷한 환자'는 거리가 가깝다")
print(f"  → 이게 KNN(Chapter 3)의 원리")
print(f"  → 임베딩(Chapter 7) = 단어를 벡터로 표현")

# ── 정리 ──────────────────────────────────
print("\n" + "="*50)
print("핵심 정리:")
print("  numpy  → 배열 연산 (모델 입력의 기본 형태)")
print("  pandas → 표(DataFrame)로 데이터 탐색/가공")
print("  matplotlib → 그래프로 시각화")
print("  shape  → (행, 열) = (샘플 수, 특성 수)")
print("  범주형 → 라벨인코딩 or 원핫인코딩으로 숫자 변환")
print("  벡터   → 데이터 1개 = 공간의 한 점")
print("="*50)
