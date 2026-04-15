# ============================================
# 3-5. 모델 비교 — 같은 데이터에 전부 적용
# 이전(3-1~3-4)과 차이: #3에 여러 모델을 넣고 비교
#
# 왜 배우는가:
#   "어떤 모델이 이 데이터에 가장 좋은가?"
#   실무에서는 항상 여러 모델을 비교한다.
#
# Ref: Coursera C2W4 Lab02
# ============================================

#0. 라이브러리 ──────────────────────────────
import numpy as np                              # numpy: 숫자 배열(행렬) 연산 라이브러리
import matplotlib.pyplot as plt                  # matplotlib.pyplot: 그래프/차트 시각화 라이브러리
from sklearn.linear_model import LogisticRegression    # LogisticRegression: 로지스틱 회귀 — 선형으로 확률을 계산해 분류하는 모델
from sklearn.tree import DecisionTreeClassifier        # DecisionTreeClassifier: 결정트리 — 질문 규칙으로 분류
from sklearn.ensemble import RandomForestClassifier    # RandomForestClassifier: 랜덤포레스트 — 트리 여러 개의 다수결
from sklearn.svm import SVC                            # SVC: 서포트벡터머신 — 마진을 최대화하는 경계로 분류
from sklearn.neighbors import KNeighborsClassifier     # KNeighborsClassifier: KNN — 가까운 K개 이웃의 다수결
from sklearn.datasets import load_iris, load_breast_cancer  # 붓꽃/유방암 예제 데이터셋
from sklearn.model_selection import train_test_split, cross_val_score  # train_test_split: 데이터 분할 / cross_val_score: 교차검증 (데이터를 여러 번 나눠서 평균 성능 측정)
from sklearn.preprocessing import StandardScaler       # StandardScaler: 정규화 (평균=0, 표준편차=1)
from sklearn.metrics import accuracy_score             # accuracy_score: 정확도 계산
import warnings                                        # warnings: 경고 메시지 제어 모듈
warnings.filterwarnings('ignore')                      # 불필요한 경고 메시지 숨기기

#1. 데이터 ─────────────────────────────────
datasets = {
    'Iris':   load_iris(),                    # 붓꽃 데이터 (150, 4) → 3클래스
    'Cancer': load_breast_cancer(),           # 유방암 데이터 (569, 30) → 2클래스
}

#2. 데이터 전처리 및 분할 ───────────────────
# (각 데이터셋마다 아래에서 split + scale)

#3. 모델 ──────────────────────────────────
models = {
    'Logistic':      LogisticRegression(max_iter=1000),               # max_iter=1000: 최대 반복 횟수 (수렴할 때까지)
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),  # max_depth=5: 트리 깊이 제한
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),  # n_estimators=100: 트리 100개
    'SVM (rbf)':     SVC(kernel='rbf'),                               # kernel='rbf': 방사형 기저함수 커널
    'KNN (k=5)':     KNeighborsClassifier(n_neighbors=5),             # n_neighbors=5: 이웃 5개
}

#4. 평가 ──────────────────────────────────
results = {}                                   # 데이터셋별 모델 성능 저장
for dname, data in datasets.items():           # 각 데이터셋에 대해 반복
    x_train, x_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42)  # 20% 테스트 분리
    scaler = StandardScaler()                  # 모든 모델에 스케일링 적용 (SVM, KNN에 필수, 나머지에도 무해)
    x_train = scaler.fit_transform(x_train)    # 훈련 데이터 기준 정규화
    x_test = scaler.transform(x_test)          # 같은 기준으로 테스트 데이터 변환

    print(f"\n{'='*55}")
    print(f"[ {dname} — {data.data.shape} ]")
    print(f"{'모델':20s} {'Accuracy':>10s} {'CV Mean':>10s}")  # CV Mean: 교차검증 평균 정확도
    print("-" * 45)

    results[dname] = {}
    for mname, model in models.items():        # 각 모델에 대해 반복
        model.fit(x_train, y_train)            # fit: 훈련 데이터로 학습
        acc = accuracy_score(y_test, model.predict(x_test))  # 테스트 정확도
        cv = cross_val_score(model, x_train, y_train, cv=5).mean()  # cross_val_score: 훈련 데이터를 5등분해서 5번 평가 후 평균 (cv=5: 5-fold 교차검증)
        results[dname][mname] = acc
        print(f"  {mname:18s} {acc:>10.4f} {cv:>10.4f}")

    best = max(results[dname].items(), key=lambda x: x[1])  # 가장 높은 정확도의 모델 찾기
    print(f"  → 최고: {best[0]} ({best[1]:.4f})")

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))  # 1x2 격자 / figsize: 가로 14, 세로 5 인치
for i, (dname, res) in enumerate(results.items()):  # 각 데이터셋별 그래프
    names = list(res.keys())                   # 모델 이름 리스트
    accs = list(res.values())                  # 정확도 리스트
    axes[i].barh(names, accs, color='steelblue', edgecolor='black')  # barh: 수평 막대그래프
    axes[i].set_title(dname)                   # 데이터셋 이름을 제목으로
    axes[i].set_xlim(0.85, 1.0)               # x축 범위 (0.85~1.0으로 차이 강조)
    for j, v in enumerate(accs):
        axes[i].text(v + 0.002, j, f'{v:.4f}', va='center', fontsize=10)  # text: 막대 옆에 정확도 숫자 표시 / va='center': 수직 가운데 정렬

plt.tight_layout()                             # 여백 자동 조정
plt.savefig('3-5_output.png', dpi=100)         # 이미지 저장
plt.show()                                     # 화면 표시

print("\n" + "="*50)
print("핵심 정리:")
print("  #3에 여러 모델을 넣고 같은 기준으로 비교")
print("  데이터에 따라 최적 모델이 다르다")
print("  항상 여러 모델 비교 → 가장 좋은 것 선택")
print("="*50)
