# ============================================
# 2-8. 과적합과 일반화 (Overfitting & Generalization)
#
# 왜 배우는가:
#   훈련 데이터에서 잘 되는데 새 데이터에서 안 되는 이유.
#   ML에서 가장 흔하고 중요한 문제.
#
# 나중에 만나는 곳:
#   → Chapter 4: EarlyStopping(4-8), Dropout
#   → 후속 챕터: KFold, cross_val_score
#
# ▶ 보고 오기: Coursera C1W3 "Overfitting"
#
# Ref: Stanford CS229 W4 / Google MLCC "Generalization"
# ============================================

import numpy as np                                    # numpy: 수학 계산 라이브러리 (배열, 랜덤, 삼각함수 등)
import matplotlib.pyplot as plt                        # matplotlib: 그래프를 그리는 라이브러리
from sklearn.preprocessing import PolynomialFeatures   # PolynomialFeatures: 다항식 특성 생성 (x → x, x², x³ ...)
from sklearn.linear_model import LinearRegression, Ridge  # LinearRegression: 선형회귀, Ridge: L2 정규화가 추가된 선형회귀
from sklearn.model_selection import train_test_split, cross_val_score, KFold  # train_test_split: 데이터 분할, cross_val_score: 교차검증, KFold: 폴드 분할 방식 제어
from sklearn.metrics import mean_squared_error         # mean_squared_error: 평균 제곱 오차 (MSE)
from sklearn.pipeline import Pipeline                  # Pipeline: 여러 처리 단계를 하나로 묶는 도구 (전처리→학습을 한번에)

# ── 1. 과적합 시연 데이터 ────────────────
np.random.seed(42)                                     # 난수 시드 고정: 실행마다 같은 결과
n = 30                                                 # 데이터 30개 생성
X = np.sort(np.random.uniform(0, 1, n)).reshape(-1, 1)  # 0~1 사이 랜덤 값 30개를 정렬 후 2D 배열로 변환
y = np.sin(2 * np.pi * X).ravel() + np.random.normal(0, 0.2, n)  # 사인 곡선 + 노이즈 → 실제 현상을 모사한 데이터

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # 70% 학습, 30% 테스트
X_plot = np.linspace(0, 1, 200).reshape(-1, 1)        # 예측선 그리기용: 0~1 사이 200개 점

# ── 2. 복잡도별 모델 비교 ────────────────
degrees = [1, 4, 15]                                   # 다항식 차수: 1차=직선(과소적합), 4차=적절, 15차=과적합
models = {}                                            # 각 모델과 결과를 저장할 딕셔너리

print("[ 모델 복잡도별 성능 비교 ]")
print(f"{'Degree':>8} {'Train MSE':>12} {'Test MSE':>12} {'상태':>10}")
print("-" * 45)

for d in degrees:                                      # 각 차수에 대해
    pipe = Pipeline([                                  # Pipeline: 다항식 변환 → 선형회귀를 순서대로 수행
        ('poly', PolynomialFeatures(degree=d)),        # 1단계: x를 x, x², ..., x^d로 확장
        ('lr', LinearRegression())                     # 2단계: 확장된 특성으로 선형회귀
    ])
    pipe.fit(X_train, y_train)                         # fit: 학습 데이터로 파이프라인 전체 학습

    train_mse = mean_squared_error(y_train, pipe.predict(X_train))  # 학습 데이터의 MSE (얼마나 잘 외웠나)
    test_mse = mean_squared_error(y_test, pipe.predict(X_test))    # 테스트 데이터의 MSE (새 데이터에서 성능)

    if d == 1:                                         # 차수 1: 직선 → 너무 단순
        status = "과소적합"
    elif d == 4:                                       # 차수 4: 곡선 → 적절한 복잡도
        status = "적절"
    else:                                              # 차수 15: 극도로 복잡 → 과적합
        status = "과적합"

    print(f"{d:>8} {train_mse:>12.4f} {test_mse:>12.4f} {status:>10}")
    models[d] = {'pipe': pipe, 'train_mse': train_mse, 'test_mse': test_mse}  # 결과 저장

# ── 3. 정규화 (Ridge) 효과 ───────────────
print(f"\n[ 정규화 (Ridge) — 과적합 방지 ]")
pipe_overfit = Pipeline([                              # 정규화 없는 15차 모델 (과적합됨)
    ('poly', PolynomialFeatures(degree=15)),           # 15차 다항식 특성 생성
    ('lr', LinearRegression())                         # 일반 선형회귀 (제약 없음)
])
pipe_overfit.fit(X_train, y_train)                     # 정규화 없는 모델 학습
test_overfit = mean_squared_error(y_test, pipe_overfit.predict(X_test))  # 정규화 없는 모델의 테스트 MSE
print(f"  Degree=15 (정규화 없음):       Test MSE = {test_overfit:.4f}")

# alpha(제약 강도)도 하이퍼파라미터 — 값을 바꿔가며 스윕해서 감을 잡는다
alphas = [0.001, 0.1, 10]                              # 약한 제약 ~ 강한 제약
ridge_pipes = {}                                       # alpha별 학습된 모델 저장
for a in alphas:                                       # 각 alpha에 대해
    pipe_r = Pipeline([                                # Ridge 정규화가 있는 15차 모델
        ('poly', PolynomialFeatures(degree=15)),       # 15차 다항식 특성 생성
        ('ridge', Ridge(alpha=a))                      # Ridge: 가중치가 너무 커지지 않도록 제약 (alpha: 제약 강도)
    ])
    pipe_r.fit(X_train, y_train)                       # Ridge 모델 학습
    ridge_pipes[a] = pipe_r                            # 저장 (시각화에 재사용)
    test_r = mean_squared_error(y_test, pipe_r.predict(X_test))  # 해당 alpha의 테스트 MSE
    print(f"  Degree=15 + Ridge(alpha={a}): Test MSE = {test_r:.4f}")
print(f"  → alpha에 따라 결과가 달라진다 (여기선 작은 alpha로도 충분, 너무 크면 과소적합 방향)")
print(f"  → alpha도 하이퍼파라미터: 문제마다 최적값이 다르므로 튜닝 대상")

pipe_ridge = ridge_pipes[0.1]                          # 시각화용: 중간 강도 alpha=0.1 모델
test_ridge = mean_squared_error(y_test, pipe_ridge.predict(X_test))  # alpha=0.1의 테스트 MSE

# ── 4. 교차검증 ──────────────────────────
print(f"\n[ 교차검증 (5-Fold, shuffle) ]")
cv = KFold(n_splits=5, shuffle=True, random_state=42)  # shuffle=True: X가 정렬되어 있어서 순서대로 자르면
# ↑ 각 fold가 한 구간(예: 0~0.2)에 몰림 → 학습 범위 밖 외삽으로 고차 다항식 MSE가 폭발. 셔플로 완화
for d in degrees:                                      # 각 차수에 대해
    pipe = Pipeline([                                  # 파이프라인 생성
        ('poly', PolynomialFeatures(degree=d)),        # 다항식 특성 변환
        ('lr', LinearRegression())                     # 선형회귀
    ])
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='neg_mean_squared_error')  # 5-Fold 교차검증: 데이터를 5등분해서 5번 학습/평가
    mean_mse = -scores.mean()                          # 결과가 음수로 나오므로 -를 붙여 양수로 변환
    std_mse = scores.std()                             # 5번 결과의 표준편차 (안정성 지표)
    print(f"  Degree={d:2d}: MSE = {mean_mse:.4f} (±{std_mse:.4f})")

# ── 5. 시각화 ─────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))        # 2행 3열 그래프 영역

# 과소적합 / 적절 / 과적합
titles = {1: 'Underfitting (d=1)', 4: 'Good Fit (d=4)', 15: 'Overfitting (d=15)'}  # 각 차수별 제목
for i, d in enumerate(degrees):                        # 각 차수에 대해
    pipe = models[d]['pipe']                           # 저장된 모델 꺼냄
    y_plot = pipe.predict(X_plot)                       # 예측선 데이터 생성

    axes[0, i].scatter(X_train, y_train, color='blue', s=30, label='Train', zorder=5)  # 학습 데이터 (파란 점)
    axes[0, i].scatter(X_test, y_test, color='red', s=30, label='Test', zorder=5)      # 테스트 데이터 (빨간 점)
    axes[0, i].plot(X_plot, y_plot, 'g-', linewidth=2, label=f'd={d}')  # 모델 예측선 (초록)
    axes[0, i].plot(X_plot, np.sin(2 * np.pi * X_plot), 'k--', alpha=0.3, label='True')  # 진짜 함수 (검은 점선)
    axes[0, i].set_title(titles[d])                    # 제목
    axes[0, i].set_ylim(-2, 2)                         # y축 범위 고정
    axes[0, i].legend(fontsize=7)                      # 범례 (글자 크기 7)

# 복잡도 vs 성능
degrees_range = range(1, 16)                           # 1차~15차까지 다양한 복잡도
train_errors = []                                      # 각 차수의 학습 에러 저장
test_errors = []                                       # 각 차수의 테스트 에러 저장
for d in degrees_range:                                # 각 차수에 대해
    pipe = Pipeline([('poly', PolynomialFeatures(degree=d)), ('lr', LinearRegression())])  # 파이프라인
    pipe.fit(X_train, y_train)                         # 학습
    train_errors.append(mean_squared_error(y_train, pipe.predict(X_train)))  # 학습 MSE 저장
    test_errors.append(mean_squared_error(y_test, pipe.predict(X_test)))     # 테스트 MSE 저장

axes[1, 0].plot(list(degrees_range), train_errors, 'b-o', markersize=4, label='Train')  # 학습 에러 곡선 (파란색, 계속 줄어듦)
axes[1, 0].plot(list(degrees_range), test_errors, 'r-o', markersize=4, label='Test')    # 테스트 에러 곡선 (빨간색, U자)
axes[1, 0].set_xlabel('Polynomial Degree')             # x축: 다항식 차수 (복잡도)
axes[1, 0].set_ylabel('MSE')                           # y축: MSE
axes[1, 0].set_title('Complexity vs Error')            # 제목: 복잡도 vs 에러
axes[1, 0].set_ylim(0, min(2, max(test_errors)))       # y축 상한 설정
axes[1, 0].axvline(4, color='green', linestyle='--', alpha=0.5, label='Sweet Spot')  # 최적 복잡도 표시
axes[1, 0].legend()                                    # 범례

# 정규화 효과
axes[1, 1].scatter(X_train, y_train, color='blue', s=30, zorder=5)  # 학습 데이터
axes[1, 1].scatter(X_test, y_test, color='red', s=30, zorder=5)     # 테스트 데이터
axes[1, 1].plot(X_plot, pipe_overfit.predict(X_plot), 'orange', linewidth=2, label='No Regularization')  # 정규화 없음 (과적합, 들쭉날쭉)
axes[1, 1].plot(X_plot, pipe_ridge.predict(X_plot), 'green', linewidth=2, label='Ridge (L2)')           # Ridge 정규화 (부드러운 곡선)
axes[1, 1].set_ylim(-2, 2)                            # y축 범위
axes[1, 1].set_title('Regularization Effect (d=15)')   # 제목: 정규화 효과
axes[1, 1].legend()                                    # 범례

# Bias-Variance 직관
labels = ['Underfitting', 'Sweet Spot', 'Overfitting']  # 세 가지 상태
bias_vals = [0.8, 0.2, 0.05]                           # Bias (편향): 모델이 단순할수록 높음 (시스템적 오류)
var_vals = [0.05, 0.2, 0.8]                            # Variance (분산): 모델이 복잡할수록 높음 (데이터에 민감)
total = [b + v for b, v in zip(bias_vals, var_vals)]   # 총 에러 = Bias + Variance
x_pos = range(len(labels))                             # x축 위치
axes[1, 2].bar(x_pos, bias_vals, 0.35, label='Bias', color='steelblue')    # Bias 막대 (아래)
axes[1, 2].bar(x_pos, var_vals, 0.35, bottom=bias_vals, label='Variance', color='salmon')  # Variance 막대 (위에 쌓기)
axes[1, 2].plot(x_pos, total, 'ko-', label='Total Error')  # 총 에러 선 (Sweet Spot에서 최소)
axes[1, 2].set_xticks(x_pos)                           # x축 눈금 위치
axes[1, 2].set_xticklabels(labels)                     # x축 눈금 라벨
axes[1, 2].set_title('Bias-Variance Tradeoff')         # 제목: 편향-분산 트레이드오프
axes[1, 2].legend()                                    # 범례

plt.tight_layout()                                     # 그래프 간 간격 자동 조정
plt.savefig('2-8_output.png', dpi=100)               # 이미지 파일로 저장
plt.show()                                             # 화면에 표시

# ── 정리 ──────────────────────────────────
print("\n" + "="*50)
print("핵심 정리:")
print("  과소적합 = 모델 너무 단순 → 레이어/특성 추가")
print("  과적합   = 모델 너무 복잡 → 정규화/데이터 추가")
print("  train/test 분할 = 반드시 나눠서 평가")
print("  교차검증 = K번 반복해서 안정적 평가")
print("  정규화 = 가중치를 크게 만들지 않는 제약 (alpha도 하이퍼파라미터)")
print("  시드 고정 = 재현성. 연구에선 multi-seed 평균±표준편차 보고가 표준")
print("="*50)
print("\n★ Chapter 2 완료! — 다음은 Chapter 3 (머신러닝) → 체크포인트 시험 1")
