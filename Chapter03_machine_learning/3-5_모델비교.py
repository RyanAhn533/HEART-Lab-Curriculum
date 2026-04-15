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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

#1. 데이터 ─────────────────────────────────
datasets = {
    'Iris':   load_iris(),
    'Cancer': load_breast_cancer(),
}

#2. 데이터 전처리 및 분할 ───────────────────
# (각 데이터셋마다 아래에서 split + scale)

#3. 모델 ──────────────────────────────────
models = {
    'Logistic':      LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM (rbf)':     SVC(kernel='rbf'),
    'KNN (k=5)':     KNeighborsClassifier(n_neighbors=5),
}

#4. 평가 ──────────────────────────────────
results = {}
for dname, data in datasets.items():
    x_train, x_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    print(f"\n{'='*55}")
    print(f"[ {dname} — {data.data.shape} ]")
    print(f"{'모델':20s} {'Accuracy':>10s} {'CV Mean':>10s}")
    print("-" * 45)

    results[dname] = {}
    for mname, model in models.items():
        model.fit(x_train, y_train)
        acc = accuracy_score(y_test, model.predict(x_test))
        cv = cross_val_score(model, x_train, y_train, cv=5).mean()
        results[dname][mname] = acc
        print(f"  {mname:18s} {acc:>10.4f} {cv:>10.4f}")

    best = max(results[dname].items(), key=lambda x: x[1])
    print(f"  → 최고: {best[0]} ({best[1]:.4f})")

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for i, (dname, res) in enumerate(results.items()):
    names = list(res.keys())
    accs = list(res.values())
    axes[i].barh(names, accs, color='steelblue', edgecolor='black')
    axes[i].set_title(dname)
    axes[i].set_xlim(0.85, 1.0)
    for j, v in enumerate(accs):
        axes[i].text(v + 0.002, j, f'{v:.4f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('3-5_output.png', dpi=100)
plt.show()

print("\n" + "="*50)
print("핵심 정리:")
print("  #3에 여러 모델을 넣고 같은 기준으로 비교")
print("  데이터에 따라 최적 모델이 다르다")
print("  항상 여러 모델 비교 → 가장 좋은 것 선택")
print("="*50)
