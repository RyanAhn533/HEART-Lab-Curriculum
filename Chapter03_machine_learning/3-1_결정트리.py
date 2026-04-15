# ============================================
# 3-1. 결정트리 (Decision Tree) — Iris
# 이전(2-8)과 차이: #3에 sklearn DecisionTreeClassifier
#
# 왜 배우는가:
#   질문을 반복해서 분류 — 사람의 사고방식과 가장 비슷.
#   해석 가능한 모델. 트리 구조를 시각화할 수 있다.
#
# 나중에 만나는 곳:
#   → 3-2 랜덤포레스트: 트리 여러 개 합치기
#   → Chapter 9 GridSearch: 하이퍼파라미터 튜닝
#
# ▶ 보고 오기: StatQuest "Decision Trees"
#
# Ref: Coursera C2W4 Lab01 / Stanford CS229
# ============================================

#0. 라이브러리 ──────────────────────────────
import numpy as np                              # numpy: 숫자 배열(행렬) 연산 라이브러리. 데이터를 효율적으로 다룰 때 사용
import matplotlib.pyplot as plt                  # matplotlib.pyplot: 그래프/차트를 그리는 시각화 라이브러리
from sklearn.tree import DecisionTreeClassifier, plot_tree  # DecisionTreeClassifier: 결정트리 분류 모델 / plot_tree: 트리 구조를 그림으로 보여주는 함수
from sklearn.datasets import load_iris           # load_iris: 붓꽃(Iris) 예제 데이터를 불러오는 함수 (꽃잎/꽃받침 길이·너비 → 품종 분류)
from sklearn.model_selection import train_test_split  # train_test_split: 데이터를 훈련용/테스트용으로 나누는 함수
from sklearn.metrics import accuracy_score, classification_report  # accuracy_score: 정확도 계산 / classification_report: 정밀도·재현율 등 상세 평가 리포트

#1. 데이터 ─────────────────────────────────
data = load_iris()            # Iris 데이터셋 로드 (150송이 붓꽃의 측정값과 품종 라벨)
x = data.data                 # x: 입력 특성 (150, 4) — 꽃받침 길이, 꽃받침 너비, 꽃잎 길이, 꽃잎 너비
y = data.target               # y: 정답 라벨 (150,) — 0(setosa), 1(versicolor), 2(virginica)

print("[ Iris — 결정트리 ]")
print(f"특성: {data.feature_names}")       # 4개 특성의 이름 출력
print(f"클래스: {list(data.target_names)}")  # 3개 품종의 이름 출력
print(f"shape: {x.shape}")                  # 데이터 크기 확인 (행 수, 열 수)

#2. 데이터 전처리 및 분할 ───────────────────
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42    # test_size=0.2: 20%를 테스트용으로 분리 / random_state=42: 랜덤 시드 고정 (재현성 보장)
)
# 결정트리는 스케일링 불필요 (트리는 크기 비교만 하므로)

#3. 모델 ──────────────────────────────────
model = DecisionTreeClassifier(max_depth=3, random_state=42)  # max_depth=3: 트리 깊이를 3으로 제한 (과적합 방지) / random_state: 재현성
model.fit(x_train, y_train)                                    # fit: 훈련 데이터로 트리를 만든다 (질문 규칙 학습)

#4. 평가 ──────────────────────────────────
y_pred = model.predict(x_test)               # predict: 학습된 트리로 새 데이터의 품종을 예측
acc = accuracy_score(y_test, y_pred)          # accuracy_score: 예측이 정답과 얼마나 일치하는지 비율로 계산 (1.0 = 100% 정확)

print(f"\n[ 평가 ]")
print(f"Accuracy: {acc:.4f}")                # 소수점 4자리까지 정확도 출력
print(f"트리 깊이: {model.get_depth()}")       # get_depth: 실제 만들어진 트리의 깊이
print(f"\n{classification_report(y_test, y_pred, target_names=data.target_names)}")  # 품종별 정밀도/재현율/F1 상세 리포트

# 특성 중요도
importance = model.feature_importances_       # feature_importances_: 각 특성이 분류에 얼마나 기여했는지 (0~1, 합=1)
print("[ 특성 중요도 ]")
for name, imp in sorted(zip(data.feature_names, importance), key=lambda x: -x[1]):  # 중요도가 높은 순으로 정렬
    bar = "█" * int(imp * 30)                 # 중요도를 막대 그래프 형태로 시각화 (텍스트)
    print(f"  {name:20s}: {imp:.3f} {bar}")

# ── 추가 분석: 깊이별 과적합 확인 ─────────
print(f"\n[ 깊이별 과적합 확인 ]")
depths = range(1, 15)                         # 트리 깊이를 1부터 14까지 바꿔가며 실험
train_accs, test_accs = [], []                # 각 깊이별 훈련/테스트 정확도를 저장할 리스트
for d in depths:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)  # 깊이 d인 결정트리 생성
    dt.fit(x_train, y_train)                                    # 훈련
    train_accs.append(accuracy_score(y_train, dt.predict(x_train)))  # 훈련 데이터 정확도 (높을수록 과적합 가능성)
    test_accs.append(accuracy_score(y_test, dt.predict(x_test)))     # 테스트 데이터 정확도 (실제 성능)

best_depth = list(depths)[np.argmax(test_accs)]  # np.argmax: 테스트 정확도가 가장 높은 깊이 찾기
print(f"최적 depth: {best_depth} (test acc={max(test_accs):.4f})")

# ── 시각화 ────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # 2x2 격자로 4개 그래프 생성 / figsize: 전체 그림 크기 (가로 16, 세로 12 인치)

# (좌상) 결정트리 구조 시각화
plot_tree(model, feature_names=data.feature_names,   # feature_names: 각 노드에 특성 이름 표시
          class_names=list(data.target_names), filled=True,  # class_names: 클래스 이름 표시 / filled: 클래스별 색칠
          rounded=True, fontsize=8, ax=axes[0, 0])    # rounded: 둥근 모서리 / fontsize: 글자 크기 / ax: 그릴 위치
axes[0, 0].set_title('Decision Tree (max_depth=3)')   # set_title: 그래프 제목

# (우상) 특성 중요도 수평 막대그래프
sorted_idx = np.argsort(importance)                   # argsort: 중요도 오름차순 정렬 인덱스
axes[0, 1].barh(np.array(data.feature_names)[sorted_idx], importance[sorted_idx], color='steelblue')  # barh: 수평 막대그래프 / color: 막대 색상
axes[0, 1].set_title('Feature Importance')

# (좌하) 깊이별 훈련/테스트 정확도 비교 (과적합 확인용)
axes[1, 0].plot(list(depths), train_accs, 'b-o', markersize=4, label='Train')   # 'b-o': 파란색 선+동그라미 / markersize: 점 크기 / label: 범례 텍스트
axes[1, 0].plot(list(depths), test_accs, 'r-o', markersize=4, label='Test')     # 'r-o': 빨간색 선+동그라미
axes[1, 0].axvline(best_depth, color='green', linestyle='--', alpha=0.5, label=f'Best={best_depth}')  # axvline: 수직선 / linestyle: 점선 / alpha: 투명도
axes[1, 0].set_xlabel('max_depth')     # set_xlabel: x축 라벨
axes[1, 0].set_ylabel('Accuracy')      # set_ylabel: y축 라벨
axes[1, 0].set_title('Depth vs Accuracy (Overfitting)')  # 깊이가 깊어지면 Train↑ Test↓ = 과적합
axes[1, 0].legend()                    # legend: 범례 표시

# (우하) 결정 경계 시각화 (2개 특성만 사용)
x_2d = x_train[:, :2]                 # 처음 2개 특성만 추출 (2D 시각화를 위해)
dt_2d = DecisionTreeClassifier(max_depth=3, random_state=42).fit(x_2d, y_train)  # 2개 특성으로 결정트리 학습
h = 0.02                              # h: 격자 간격 (작을수록 결정 경계가 부드러움)
xx, yy = np.meshgrid(np.arange(x_2d[:,0].min()-0.5, x_2d[:,0].max()+0.5, h),    # meshgrid: 2D 격자 좌표 생성 (결정 경계를 그리기 위함)
                      np.arange(x_2d[:,1].min()-0.5, x_2d[:,1].max()+0.5, h))
Z = dt_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)  # 격자의 모든 점을 예측 → 결정 경계 생성 / ravel: 1차원으로 펼침 / reshape: 원래 격자 모양으로 복원
axes[1, 1].contourf(xx, yy, Z, alpha=0.3, cmap='viridis')           # contourf: 등고선 형태로 결정 영역 색칠 / cmap: 색상맵 / alpha: 투명도
axes[1, 1].scatter(x_2d[:, 0], x_2d[:, 1], c=y_train, cmap='viridis', s=20, edgecolors='k')  # scatter: 산점도 / c: 색상(라벨) / s: 점 크기 / edgecolors: 테두리 색
axes[1, 1].set_title('Decision Boundary (2 features)')

plt.tight_layout()                     # tight_layout: 그래프 간 여백 자동 조정
plt.savefig('3-1_output.png', dpi=100) # savefig: 그래프를 이미지 파일로 저장 / dpi: 해상도 (100 = 보통)
plt.show()                             # show: 화면에 그래프 표시

# ── 정리 ──────────────────────────────────
print("\n" + "="*50)
print("핵심 정리:")
print("  #3만 바뀜: DecisionTreeClassifier(max_depth=3)")
print("  스케일링 불필요 (트리는 크기 비교만)")
print("  max_depth = 과적합 방지 핵심")
print("  장점: 해석 쉬움, 시각화 가능")
print("  단점: 과적합 → 랜덤포레스트로 해결 (3-2)")
print("="*50)
