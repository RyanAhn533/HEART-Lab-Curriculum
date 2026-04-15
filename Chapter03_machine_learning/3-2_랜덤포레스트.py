# ============================================
# 3-2. 랜덤포레스트 (Random Forest) — Cancer
# 이전(3-1)과 차이: #3에 RandomForestClassifier (트리 여러 개)
#
# 왜 배우는가:
#   트리 1개는 불안정 → 100개 합치면 안정적.
#   테이블 데이터에서 가장 강력한 모델 중 하나.
#
# ▶ 보고 오기: StatQuest "Random Forests"
#
# Ref: Coursera C2W4 Lab02 / AI5 ml/m37
# ============================================

#0. 라이브러리 ──────────────────────────────
import numpy as np                              # numpy: 숫자 배열(행렬) 연산 라이브러리
import matplotlib.pyplot as plt                  # matplotlib.pyplot: 그래프/차트 시각화 라이브러리
from sklearn.tree import DecisionTreeClassifier  # DecisionTreeClassifier: 단일 결정트리 (비교용)
from sklearn.ensemble import RandomForestClassifier  # RandomForestClassifier: 랜덤포레스트 — 결정트리 여러 개를 합쳐서 투표하는 앙상블 모델
from sklearn.datasets import load_breast_cancer  # load_breast_cancer: 유방암 데이터셋 (30개 특성 → 악성/양성 분류)
from sklearn.model_selection import train_test_split  # train_test_split: 데이터를 훈련/테스트로 분할
from sklearn.metrics import accuracy_score, classification_report  # accuracy_score: 정확도 / classification_report: 상세 평가

#1. 데이터 ─────────────────────────────────
data = load_breast_cancer()   # 유방암 데이터 로드 (569명의 종양 측정값)
x = data.data                 # x: 입력 특성 (569, 30) — 종양의 반지름, 질감, 둘레 등 30가지 측정값
y = data.target               # y: 정답 라벨 (569,) — 0(악성), 1(양성)

print(f"[ Breast Cancer — 랜덤포레스트 ]")
print(f"shape: {x.shape}")    # 데이터 크기 확인

#2. 데이터 전처리 및 분할 ───────────────────
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42    # 20% 테스트용 분리 / random_state: 재현성
)
# 트리 계열은 스케일링 불필요

#3. 모델 ──────────────────────────────────
model = RandomForestClassifier(                      # ← NEW: 트리 100개 앙상블
    n_estimators=100,           # n_estimators=100: 트리 100개를 만들어서 다수결 투표 (많을수록 안정적, 100이면 충분)
    max_depth=5,                # max_depth=5: 각 트리의 최대 깊이 제한 (과적합 방지)
    random_state=42,            # random_state: 랜덤 시드 고정 (재현성)
)
model.fit(x_train, y_train)     # fit: 100개의 결정트리를 각각 학습시킨다 (각 트리는 데이터와 특성을 랜덤으로 뽑아 사용)

# 비교용: 단일 결정트리
dt = DecisionTreeClassifier(max_depth=5, random_state=42)  # 같은 깊이의 트리 1개 (RF와 비교하기 위함)
dt.fit(x_train, y_train)       # 단일 트리 학습

#4. 평가 ──────────────────────────────────
y_pred = model.predict(x_test)                   # predict: 100개 트리가 각각 예측 → 다수결로 최종 결정
acc_rf = accuracy_score(y_test, y_pred)           # 랜덤포레스트 정확도
acc_dt = accuracy_score(y_test, dt.predict(x_test))  # 단일 결정트리 정확도

print(f"\n[ 평가 ]")
print(f"Decision Tree:   {acc_dt:.4f}")           # 단일 트리 성능
print(f"Random Forest:   {acc_rf:.4f}")           # 랜덤포레스트 성능 (보통 더 높음)
print(f"→ RF가 {'더 좋음' if acc_rf > acc_dt else '비슷'}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Malignant', 'Benign'])}")  # Malignant=악성, Benign=양성

# 특성 중요도 (상위 10개)
importance = model.feature_importances_           # feature_importances_: 각 특성이 분류에 얼마나 기여했는지 (100개 트리의 평균)
top_idx = np.argsort(importance)[-10:]            # argsort: 중요도 오름차순 정렬 → [-10:]: 상위 10개 인덱스
print("[ 특성 중요도 상위 10 ]")
for idx in reversed(top_idx):                     # reversed: 높은 순서대로 출력
    bar = "█" * int(importance[idx] * 50)          # 텍스트 막대그래프
    print(f"  {data.feature_names[idx]:25s}: {importance[idx]:.3f} {bar}")

# n_estimators별 성능
print(f"\n[ 트리 개수별 성능 ]")
for n in [1, 10, 50, 100, 200]:                   # 트리 개수를 바꿔가며 성능 확인
    rf = RandomForestClassifier(n_estimators=n, random_state=42).fit(x_train, y_train)  # n개 트리로 학습
    print(f"  n={n:4d}: {accuracy_score(y_test, rf.predict(x_test)):.4f}")  # 트리가 많을수록 성능↑ (100쯤이면 수렴)

# XGBoost (있으면)
try:
    from xgboost import XGBClassifier             # XGBClassifier: XGBoost — 랜덤포레스트의 발전형 (부스팅 방식), 실무에서 가장 많이 쓰임
    xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,  # learning_rate=0.1: 학습률 — 한 번에 얼마나 보정할지 (작을수록 신중)
                         random_state=42, verbosity=0)  # verbosity=0: 학습 중 로그 출력 끄기
    xgb.fit(x_train, y_train)                      # XGBoost 학습
    acc_xgb = accuracy_score(y_test, xgb.predict(x_test))
    print(f"\n[ XGBoost: {acc_xgb:.4f} ]")
except ImportError:
    print(f"\n[ XGBoost 미설치 — pip install xgboost ]")

# ── 시각화 ────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1x3 격자로 3개 그래프 / figsize: 가로 15, 세로 5 인치

# (좌) 단일트리 vs 랜덤포레스트 정확도 비교
axes[0].bar(['Decision Tree', 'Random Forest'], [acc_dt, acc_rf],
            color=['salmon', 'steelblue'], edgecolor='black')  # bar: 막대그래프 / color: 막대 색상 / edgecolor: 테두리 색
axes[0].set_title('DT vs RF')        # 그래프 제목
axes[0].set_ylim(0.9, 1.0)           # set_ylim: y축 범위 (0.9~1.0으로 차이를 강조)

# (중) 특성 중요도 상위 10개
axes[1].barh(np.array(data.feature_names)[top_idx], importance[top_idx], color='steelblue')  # barh: 수평 막대그래프
axes[1].set_title('Top 10 Feature Importance')

# (우) 단일 트리 50개의 성능 분포 vs RF
single_accs = [accuracy_score(y_test,
    DecisionTreeClassifier(max_depth=5, random_state=i).fit(x_train, y_train).predict(x_test))  # 랜덤시드를 바꿔가며 트리 50개 각각의 성능 측정
    for i in range(50)]
axes[2].hist(single_accs, bins=15, alpha=0.7, edgecolor='black', label='Single Trees')  # hist: 히스토그램 / bins: 막대 개수 / alpha: 투명도
axes[2].axvline(acc_rf, color='red', linewidth=2, linestyle='--', label=f'RF={acc_rf:.4f}')  # axvline: 수직선으로 RF 성능 표시
axes[2].set_title('Single Trees vs RF')  # 개별 트리는 들쭉날쭉, RF는 안정적
axes[2].legend()                         # legend: 범례 표시

plt.tight_layout()                       # 그래프 간 여백 자동 조정
plt.savefig('3-2_output.png', dpi=100)   # 이미지 파일로 저장
plt.show()                               # 화면에 표시

# ── 정리 ──────────────────────────────────
print("\n" + "="*50)
print("핵심 정리:")
print("  #3만 바뀜: RandomForestClassifier(n_estimators=100)")
print("  = 트리 100개 + 투표 (Bagging + 특성 랜덤)")
print("  n_estimators ↑ → 성능 ↑ (100이면 충분)")
print("  테이블 데이터 최강 → XGBoost도 같은 계열")
print("="*50)
