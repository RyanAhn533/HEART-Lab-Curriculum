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

import numpy as np                # numpy: 수학 계산 라이브러리 (배열, 난수, 통계 등)
import pandas as pd               # pandas: 표(테이블) 형태 데이터를 다루는 라이브러리
import matplotlib.pyplot as plt   # matplotlib: 그래프/차트를 그리는 라이브러리
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
# ↑ LabelEncoder: 범주형 문자열을 숫자(0,1,2...)로 변환
# ↑ StandardScaler: 평균=0, 표준편차=1로 스케일링 (가장 범용적)
# ↑ MinMaxScaler: 최솟값=0, 최댓값=1 범위로 스케일링
# ↑ RobustScaler: 중앙값과 IQR 기준 스케일링 (이상치에 강함)

# ── 1. 인코딩 ─────────────────────────────
print("[ 인코딩 — 범주형을 숫자로 ]")

df = pd.DataFrame({               # 직원 5명의 데이터를 표로 생성
    'name': ['Kim', 'Lee', 'Park', 'Choi', 'Jung'],          # 이름
    'department': ['Sales', 'Engineering', 'HR', 'Sales', 'Engineering'],  # 부서 (범주형)
    'salary': [45000, 62000, 38000, 47000, 58000],            # 연봉 (숫자형)
})
print("원본:")
print(df)

# Label Encoding — 범주를 숫자 하나로 변환 (예: Engineering=0, HR=1, Sales=2)
le = LabelEncoder()               # LabelEncoder 객체 생성
df['dept_label'] = le.fit_transform(df['department'])
# ↑ fit_transform: 범주 학습(fit) + 숫자로 변환(transform)을 한 번에
# ↑ 알파벳 순으로 번호 부여: Engineering=0, HR=1, Sales=2
print(f"\nLabel Encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")
# ↑ classes_: 학습된 범주 목록. zip으로 범주와 번호를 딕셔너리로 묶어 출력

# One-Hot Encoding — 범주를 0/1 열 여러 개로 변환 (순서가 없는 범주에 적합)
df_onehot = pd.get_dummies(df[['department']], prefix='dept')
# ↑ get_dummies: 각 범주를 별도 열로 만듦. 해당하면 1, 아니면 0
# ↑ prefix='dept'→열 이름 앞에 'dept_' 접두사 추가
print(f"\nOne-Hot Encoding:")
print(pd.concat([df[['name', 'department']], df_onehot], axis=1))
# ↑ concat: 이름/부서와 One-Hot 열을 옆으로(axis=1) 붙여서 보여줌

# ── 2. 스케일링 비교 ─────────────────────
print("\n\n[ 스케일링 — 범위 맞추기 ]")

np.random.seed(42)                # 랜덤 시드 고정
data = pd.DataFrame({
    'height': np.random.normal(170, 7, 100),       # 키: 평균 170cm, 표준편차 7cm → 대략 155~185 범위
    'weight': np.random.normal(68, 10, 100),        # 몸무게: 평균 68kg, 표준편차 10kg → 대략 40~100 범위
    'income': np.random.exponential(40000, 100),    # 소득: 지수분포(치우침) → 0~200000 범위
})

# 이상치 추가 — 스케일러별 이상치 대응력 차이를 보기 위해
data.loc[0, 'income'] = 500000    # 첫 번째 행의 income을 50만으로 설정 (극단적 이상치)

print("원본 데이터 범위:")
for col in data.columns:          # 각 열에 대해 반복
    print(f"  {col}: {data[col].min():.0f} ~ {data[col].max():.0f} (mean={data[col].mean():.0f})")
    # ↑ 열별 최솟값, 최댓값, 평균 출력 — 범위 차이가 크다는 것을 확인

scalers = {                       # 비교할 스케일러 3종을 딕셔너리로 정의
    'MinMaxScaler': MinMaxScaler(),     # 0~1 범위로 변환. 이상치에 민감
    'StandardScaler': StandardScaler(), # 평균=0, 표준편차=1로 변환. 가장 범용적
    'RobustScaler': RobustScaler(),     # 중앙값=0, IQR=1 기준 변환. 이상치에 강함
}

results = {}                      # 스케일링 결과를 저장할 딕셔너리
for name, scaler in scalers.items():  # 3개 스케일러를 순서대로 실행
    scaled = scaler.fit_transform(data)  # fit_transform: 데이터 기준 학습 + 변환을 한 번에
    results[name] = pd.DataFrame(scaled, columns=data.columns)  # 결과를 DataFrame으로 저장
    print(f"\n{name} 후:")
    for col in data.columns:
        print(f"  {col}: {results[name][col].min():.2f} ~ {results[name][col].max():.2f} (mean={results[name][col].mean():.2f})")
        # ↑ 스케일링 후 각 열의 범위와 평균 출력 — 세 스케일러의 차이 비교

# ── 3. fit/transform 분리 (핵심!) ────────
print("\n\n[ fit/transform 분리 — Data Leakage 방지 ]")
from sklearn.model_selection import train_test_split  # 데이터를 학습용/테스트용으로 나누는 함수

X_train, X_test = train_test_split(data, test_size=0.2, random_state=42)
# ↑ test_size=0.2→20%는 테스트, 80%는 학습용. random_state=재현성 보장
scaler = StandardScaler()         # StandardScaler 사용

# 올바른 방법 — train 데이터만으로 기준을 정하고, test도 같은 기준으로 변환
scaler.fit(X_train)                       # fit: train 데이터의 평균/표준편차를 학습 (기준 설정)
X_train_scaled = scaler.transform(X_train)  # transform: 학습한 기준으로 train 데이터 변환
X_test_scaled = scaler.transform(X_test)    # transform: 같은 기준으로 test 데이터 변환
# ↑ 핵심: test 데이터로 fit하면 안 됨! (시험 답을 미리 보는 것 = Data Leakage)

print(f"Train mean (scaled): {X_train_scaled.mean(axis=0).round(2)}")
# ↑ axis=0→열별 평균. train은 fit 기준이므로 평균이 거의 0
print(f"Test mean (scaled):  {X_test_scaled.mean(axis=0).round(2)}")
# ↑ test는 다른 데이터이므로 평균이 0이 아닐 수 있음 — 이게 정상
print("→ Train은 0에 가깝고, Test는 약간 다름 = 정상")

# ── 4. 시각화 ─────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))  # 2행 3열 = 6개 차트, figsize=(가로 15인치, 세로 8인치)

# 원본 분포 — 3개 변수의 원본 히스토그램
for i, col in enumerate(data.columns):  # enumerate: 인덱스(i)와 열 이름(col)을 동시에 가져옴
    axes[0, i].hist(data[col], bins=20, edgecolor='black', alpha=0.7)
    # ↑ 각 변수의 히스토그램. bins=막대 20개
    axes[0, i].set_title(f'Original: {col}')  # 제목: 원본 변수명

# 스케일링 비교 (income 열) — 3개 스케일러별 income 분포 비교
colors = ['blue', 'green', 'orange']  # 각 스케일러의 막대 색상
for i, (name, result) in enumerate(results.items()):  # 3개 스케일러 결과를 순회
    axes[1, i].hist(result['income'], bins=20, edgecolor='black', alpha=0.7, color=colors[i])
    # ↑ 스케일링된 income의 히스토그램
    axes[1, i].set_title(f'{name}: income')  # 제목: 스케일러 이름 + income

plt.tight_layout()               # 차트 간 간격 자동 조정
plt.savefig('1-5_output.png', dpi=100)  # 이미지 파일로 저장. dpi=해상도
plt.show()                       # 화면에 그래프 표시

# ── 정리 ──────────────────────────────────
print("\n" + "="*50)             # 구분선 출력
print("핵심 정리:")
print("  Label Encoding → 트리 계열, 순서 있는 범주")
print("  One-Hot → 신경망, 회귀, 순서 없는 범주")
print("  MinMaxScaler → 0~1 범위, 이상치 민감")
print("  StandardScaler → 평균0 표준편차1, 가장 범용적")
print("  RobustScaler → 중앙값/IQR 기준, 이상치에 강함")
print("  fit은 train만! transform은 train+test!")
print("="*50)
