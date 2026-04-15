# ============================================
# 3-8. 전통 ML의 한계 → 딥러닝이 필요한 이유
#
# 왜 배우는가:
#   Chapter 3에서 배운 모델들(트리, RF, SVM, KNN)을
#   이미지 데이터에 적용하면 한계가 보인다.
#   → "그래서 딥러닝(Chapter 4)이 필요하다"
#
# 이 파일이 Chapter 3 → Chapter 4의 다리 역할.
# ============================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist
import warnings
warnings.filterwarnings('ignore')

# ── 1. MNIST 로드 ────────────────────────
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Flatten (28x28 → 784)
x_train_flat = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test_flat = x_test.reshape(x_test.shape[0], -1) / 255.0

# 학습 속도를 위해 일부만 사용
n_train = 5000
n_test = 1000
x_tr = x_train_flat[:n_train]
y_tr = y_train[:n_train]
x_te = x_test_flat[:n_test]
y_te = y_test[:n_test]

print(f"[ MNIST — 전통 ML 모델 비교 ]")
print(f"Train: {x_tr.shape}, Test: {x_te.shape}")
print(f"(전체 60000개 중 {n_train}개만 사용 — 속도)")

# ── 2. 전통 ML 모델들 적용 ───────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree':       DecisionTreeClassifier(max_depth=10, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM (rbf)':           SVC(kernel='rbf'),
    'KNN (k=3)':           KNeighborsClassifier(n_neighbors=3),
}

results = {}
print(f"\n{'모델':25s} {'Accuracy':>10s} {'한계':>20s}")
print("-" * 58)

for name, model in models.items():
    model.fit(x_tr, y_tr)
    acc = accuracy_score(y_te, model.predict(x_te))
    results[name] = acc

    if acc < 0.92:
        limit = "부족"
    elif acc < 0.96:
        limit = "괜찮지만..."
    else:
        limit = "좋음"
    print(f"  {name:23s} {acc:>10.4f} {limit:>20s}")

# ── 3. 한계 분석 ─────────────────────────
print(f"\n{'='*58}")
print("[ 전통 ML의 한계 ]")
print(f"{'='*58}")
print(f"1. 이미지를 Flatten(784D)하면 공간 정보가 사라진다")
print(f"   → 28x28 이미지의 '위치 관계'를 모른다")
print(f"   → CNN은 이 공간 정보를 보존한다 (Chapter 5)")
print(f"")
print(f"2. 특성을 사람이 만들어야 한다 (Feature Engineering)")
print(f"   → 딥러닝은 특성을 자동으로 학습한다")
print(f"")
print(f"3. 데이터가 많아질수록 전통 ML 성능이 정체된다")
print(f"   → 딥러닝은 데이터가 많을수록 성능이 올라간다")
print(f"")
print(f"4. 텍스트, 음성 같은 비정형 데이터는 더 어렵다")
print(f"   → RNN, Transformer가 필요한 이유 (Chapter 6~7)")

# ── 4. DNN과 비교 (맛보기) ───────────────
print(f"\n[ 참고: DNN (Chapter 4에서 배울 것) ]")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

model_dnn = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax'),
])
model_dnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_dnn.fit(x_tr, to_categorical(y_tr, 10), epochs=10, batch_size=32, verbose=0)
_, acc_dnn = model_dnn.evaluate(x_te, to_categorical(y_te, 10), verbose=0)
results['DNN (Chapter 4)'] = acc_dnn
print(f"  DNN accuracy: {acc_dnn:.4f}")
print(f"  → 같은 데이터, 같은 Flatten인데 DNN이 더 좋다")
print(f"  → CNN(Chapter 5)으로 바꾸면 99%+ 가능")

# ── 5. 시각화 ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 모델별 성능 비교
colors = ['salmon'] * 5 + ['steelblue']
names = list(results.keys())
accs = list(results.values())
bars = axes[0].barh(names, accs, color=colors, edgecolor='black')
axes[0].set_xlabel('Accuracy')
axes[0].set_title('MNIST: Traditional ML vs DNN')
axes[0].set_xlim(0.8, 1.0)
for bar, acc in zip(bars, accs):
    axes[0].text(acc + 0.002, bar.get_y() + bar.get_height()/2,
                 f'{acc:.4f}', va='center', fontsize=10)

# MNIST 샘플 이미지
for i in range(10):
    ax_sub = axes[1].inset_axes([0.1 * i, 0, 0.1, 1])
    ax_sub.imshow(x_train[i], cmap='gray')
    ax_sub.set_title(str(y_train[i]), fontsize=10)
    ax_sub.axis('off')
axes[1].axis('off')
axes[1].set_title('MNIST Samples (28x28 images → Flatten to 784D)')

plt.tight_layout()
plt.savefig('3-8_output.png', dpi=100)
plt.show()

# ── 정리 ──────────────────────────────────
print("\n" + "="*55)
print("핵심 정리:")
print("  전통 ML은 테이블 데이터에서 강하다")
print("  하지만 이미지/텍스트/음성에서는 한계")
print("  → 공간 정보 손실 (Flatten)")
print("  → 특성을 수동으로 만들어야 함")
print("  → 데이터 많아져도 성능 정체")
print("")
print("  Chapter 4: 딥러닝 → 특성 자동 학습")
print("  Chapter 5: CNN → 공간 정보 보존")
print("  Chapter 6: RNN → 시간 정보 보존")
print("="*55)
print("\n★ Chapter 3 완료! → 체크포인트 시험 1 (Chapter 1~3)")
