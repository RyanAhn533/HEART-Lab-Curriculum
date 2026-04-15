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

import numpy as np                              # numpy: 숫자 배열(행렬) 연산 라이브러리
import matplotlib.pyplot as plt                  # matplotlib.pyplot: 그래프/차트 시각화 라이브러리
from sklearn.ensemble import RandomForestClassifier  # RandomForestClassifier: 랜덤포레스트 — 트리 여러 개의 다수결
from sklearn.svm import SVC                      # SVC: 서포트벡터머신 — 마진 최대화 분류
from sklearn.neighbors import KNeighborsClassifier   # KNeighborsClassifier: KNN — 가까운 이웃의 다수결
from sklearn.linear_model import LogisticRegression  # LogisticRegression: 로지스틱 회귀 — 선형 확률 분류
from sklearn.tree import DecisionTreeClassifier  # DecisionTreeClassifier: 결정트리 — 질문 규칙 분류
from sklearn.preprocessing import StandardScaler # StandardScaler: 정규화 (평균=0, 표준편차=1)
from sklearn.metrics import accuracy_score       # accuracy_score: 정확도 계산
from tensorflow.keras.datasets import mnist      # mnist: 손글씨 숫자(0~9) 이미지 데이터셋 (28x28 픽셀, 6만장 훈련 + 1만장 테스트)
import warnings                                  # warnings: 경고 메시지 제어
warnings.filterwarnings('ignore')                # 불필요한 경고 숨기기

# ── 1. MNIST 로드 ────────────────────────
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # MNIST 데이터 로드: 훈련 6만장, 테스트 1만장의 손글씨 숫자 이미지

# Flatten (28x28 → 784)
x_train_flat = x_train.reshape(x_train.shape[0], -1) / 255.0  # reshape: 28x28 이미지를 784 길이의 1차원 벡터로 펼침 (ML 모델은 2D 이미지를 못 읽으니까) / /255.0: 픽셀값 0~255를 0~1로 정규화
x_test_flat = x_test.reshape(x_test.shape[0], -1) / 255.0     # 테스트 데이터도 같은 방식으로 변환

# 학습 속도를 위해 일부만 사용
n_train = 5000                                # 전체 6만장 중 5000장만 사용 (전통 ML은 느리니까)
n_test = 1000                                 # 테스트도 1000장만
x_tr = x_train_flat[:n_train]                 # 훈련 데이터 5000장
y_tr = y_train[:n_train]                      # 훈련 라벨
x_te = x_test_flat[:n_test]                   # 테스트 데이터 1000장
y_te = y_test[:n_test]                        # 테스트 라벨

print(f"[ MNIST — 전통 ML 모델 비교 ]")
print(f"Train: {x_tr.shape}, Test: {x_te.shape}")  # (5000, 784), (1000, 784)
print(f"(전체 60000개 중 {n_train}개만 사용 — 속도)")

# ── 2. 전통 ML 모델들 적용 ───────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),               # max_iter=1000: 최대 반복 횟수
    'Decision Tree':       DecisionTreeClassifier(max_depth=10, random_state=42),  # max_depth=10: 트리 깊이 제한
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),  # n_estimators=100: 트리 100개
    'SVM (rbf)':           SVC(kernel='rbf'),                               # kernel='rbf': 방사형 기저함수 커널
    'KNN (k=3)':           KNeighborsClassifier(n_neighbors=3),             # n_neighbors=3: 가까운 3개 이웃
}

results = {}                                   # 모델별 정확도 저장
print(f"\n{'모델':25s} {'Accuracy':>10s} {'한계':>20s}")
print("-" * 58)

for name, model in models.items():             # 각 모델에 대해 반복
    model.fit(x_tr, y_tr)                       # fit: 5000장으로 학습
    acc = accuracy_score(y_te, model.predict(x_te))  # 1000장으로 테스트
    results[name] = acc

    if acc < 0.92:                              # 정확도에 따라 한계 수준 표시
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
from tensorflow.keras.models import Sequential   # Sequential: 층(layer)을 순서대로 쌓는 가장 기본적인 딥러닝 모델 구조
from tensorflow.keras.layers import Dense         # Dense: 완전연결층 — 이전 층의 모든 뉴런과 연결되는 층
from tensorflow.keras.utils import to_categorical # to_categorical: 정수 라벨(3)을 원핫벡터([0,0,0,1,0,...])로 변환 (다중분류에 필요)

model_dnn = Sequential([                         # Sequential: 층을 순서대로 쌓아서 모델 구성
    Dense(128, activation='relu', input_shape=(784,)),  # Dense(128): 뉴런 128개짜리 층 / activation='relu': 활성화 함수 (음수→0, 양수→그대로) / input_shape: 입력 크기 784
    Dense(64, activation='relu'),                # Dense(64): 뉴런 64개짜리 은닉층
    Dense(10, activation='softmax'),             # Dense(10): 출력층 10개 (숫자 0~9) / softmax: 각 클래스의 확률로 변환 (합=1)
])
model_dnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # compile: 학습 설정 / loss: 손실함수(다중분류용) / optimizer='adam': 경사하강법 최적화기 / metrics: 평가지표
model_dnn.fit(x_tr, to_categorical(y_tr, 10), epochs=10, batch_size=32, verbose=0)  # fit: 학습 / epochs=10: 전체 데이터 10번 반복 / batch_size=32: 32개씩 묶어서 학습 / verbose=0: 로그 숨김
_, acc_dnn = model_dnn.evaluate(x_te, to_categorical(y_te, 10), verbose=0)  # evaluate: 테스트 데이터로 성능 평가 / _: 손실값 (사용 안 함)
results['DNN (Chapter 4)'] = acc_dnn
print(f"  DNN accuracy: {acc_dnn:.4f}")
print(f"  → 같은 데이터, 같은 Flatten인데 DNN이 더 좋다")
print(f"  → CNN(Chapter 5)으로 바꾸면 99%+ 가능")

# ── 5. 시각화 ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))  # 1x2 격자 / figsize: 가로 14, 세로 5 인치

# (좌) 모델별 성능 비교 막대그래프
colors = ['salmon'] * 5 + ['steelblue']       # 전통 ML은 빨간색, DNN은 파란색으로 구분
names = list(results.keys())                   # 모델 이름 리스트
accs = list(results.values())                  # 정확도 리스트
bars = axes[0].barh(names, accs, color=colors, edgecolor='black')  # barh: 수평 막대그래프
axes[0].set_xlabel('Accuracy')                 # x축: 정확도
axes[0].set_title('MNIST: Traditional ML vs DNN')
axes[0].set_xlim(0.8, 1.0)                    # x축 범위 (0.8~1.0)
for bar, acc in zip(bars, accs):               # 각 막대 옆에 정확도 숫자 표시
    axes[0].text(acc + 0.002, bar.get_y() + bar.get_height()/2,  # text: 텍스트 표시 / get_y: 막대 y좌표 / get_height: 막대 높이
                 f'{acc:.4f}', va='center', fontsize=10)  # va='center': 수직 가운데 정렬

# (우) MNIST 샘플 이미지 10개 표시
for i in range(10):
    ax_sub = axes[1].inset_axes([0.1 * i, 0, 0.1, 1])  # inset_axes: 그래프 안에 작은 그래프 삽입 [x위치, y위치, 너비, 높이] (0~1 비율)
    ax_sub.imshow(x_train[i], cmap='gray')     # imshow: 이미지 표시 / cmap='gray': 흑백
    ax_sub.set_title(str(y_train[i]), fontsize=10)  # 이미지 위에 정답 숫자 표시
    ax_sub.axis('off')                          # axis('off'): 축 눈금 숨기기
axes[1].axis('off')                             # 바깥 축도 숨기기
axes[1].set_title('MNIST Samples (28x28 images → Flatten to 784D)')

plt.tight_layout()                              # 여백 자동 조정
plt.savefig('3-8_output.png', dpi=100)          # 이미지 저장
plt.show()                                      # 화면 표시

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
