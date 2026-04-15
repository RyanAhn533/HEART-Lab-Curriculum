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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

#1. 데이터 ─────────────────────────────────
data = load_iris()
x = data.data        # (150, 4)
y = data.target       # (150,)  — 0, 1, 2

print("[ Iris — 결정트리 ]")
print(f"특성: {data.feature_names}")
print(f"클래스: {list(data.target_names)}")
print(f"shape: {x.shape}")

#2. 데이터 전처리 및 분할 ───────────────────
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
# 결정트리는 스케일링 불필요 (트리는 크기 비교만 하므로)

#3. 모델 ──────────────────────────────────
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(x_train, y_train)

#4. 평가 ──────────────────────────────────
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n[ 평가 ]")
print(f"Accuracy: {acc:.4f}")
print(f"트리 깊이: {model.get_depth()}")
print(f"\n{classification_report(y_test, y_pred, target_names=data.target_names)}")

# 특성 중요도
importance = model.feature_importances_
print("[ 특성 중요도 ]")
for name, imp in sorted(zip(data.feature_names, importance), key=lambda x: -x[1]):
    bar = "█" * int(imp * 30)
    print(f"  {name:20s}: {imp:.3f} {bar}")

# ── 추가 분석: 깊이별 과적합 확인 ─────────
print(f"\n[ 깊이별 과적합 확인 ]")
depths = range(1, 15)
train_accs, test_accs = [], []
for d in depths:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt.fit(x_train, y_train)
    train_accs.append(accuracy_score(y_train, dt.predict(x_train)))
    test_accs.append(accuracy_score(y_test, dt.predict(x_test)))

best_depth = list(depths)[np.argmax(test_accs)]
print(f"최적 depth: {best_depth} (test acc={max(test_accs):.4f})")

# ── 시각화 ────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

plot_tree(model, feature_names=data.feature_names,
          class_names=list(data.target_names), filled=True,
          rounded=True, fontsize=8, ax=axes[0, 0])
axes[0, 0].set_title('Decision Tree (max_depth=3)')

sorted_idx = np.argsort(importance)
axes[0, 1].barh(np.array(data.feature_names)[sorted_idx], importance[sorted_idx], color='steelblue')
axes[0, 1].set_title('Feature Importance')

axes[1, 0].plot(list(depths), train_accs, 'b-o', markersize=4, label='Train')
axes[1, 0].plot(list(depths), test_accs, 'r-o', markersize=4, label='Test')
axes[1, 0].axvline(best_depth, color='green', linestyle='--', alpha=0.5, label=f'Best={best_depth}')
axes[1, 0].set_xlabel('max_depth')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].set_title('Depth vs Accuracy (Overfitting)')
axes[1, 0].legend()

x_2d = x_train[:, :2]
dt_2d = DecisionTreeClassifier(max_depth=3, random_state=42).fit(x_2d, y_train)
h = 0.02
xx, yy = np.meshgrid(np.arange(x_2d[:,0].min()-0.5, x_2d[:,0].max()+0.5, h),
                      np.arange(x_2d[:,1].min()-0.5, x_2d[:,1].max()+0.5, h))
Z = dt_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
axes[1, 1].contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
axes[1, 1].scatter(x_2d[:, 0], x_2d[:, 1], c=y_train, cmap='viridis', s=20, edgecolors='k')
axes[1, 1].set_title('Decision Boundary (2 features)')

plt.tight_layout()
plt.savefig('3-1_output.png', dpi=100)
plt.show()

# ── 정리 ──────────────────────────────────
print("\n" + "="*50)
print("핵심 정리:")
print("  #3만 바뀜: DecisionTreeClassifier(max_depth=3)")
print("  스케일링 불필요 (트리는 크기 비교만)")
print("  max_depth = 과적합 방지 핵심")
print("  장점: 해석 쉬움, 시각화 가능")
print("  단점: 과적합 → 랜덤포레스트로 해결 (3-2)")
print("="*50)
