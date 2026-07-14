# ============================================
# 2-7. 평가지표 (Evaluation Metrics)
#
# 왜 배우는가:
#   "모델이 잘 했는가?"를 판단하는 기준.
#   Accuracy만 보면 큰 실수를 할 수 있다.
#
# 나중에 만나는 곳:
#   → Chapter 4: 분류 모델 평가
#   → 논문 결과표: Accuracy, F1, AUC
#
# ▶ 보고 오기: StatQuest "ROC and AUC"
#
# Ref: Google MLCC "Classification" / Coursera C2W3
# ============================================

import numpy as np                                    # numpy: 수학 계산 라이브러리 (배열, 랜덤 등)
import matplotlib.pyplot as plt                        # matplotlib: 그래프를 그리는 라이브러리
from sklearn.datasets import load_breast_cancer        # load_breast_cancer: 유방암 예제 데이터셋
from sklearn.linear_model import LogisticRegression    # LogisticRegression: 분류 모델
from sklearn.model_selection import train_test_split   # train_test_split: 데이터를 학습/테스트로 분할
from sklearn.preprocessing import StandardScaler       # StandardScaler: 특성값 정규화 (평균0, 표준편차1)
from sklearn.metrics import (confusion_matrix, classification_report,  # confusion_matrix: 혼동행렬, classification_report: 종합 보고서
                              accuracy_score, precision_score, recall_score,  # accuracy: 정확도, precision: 정밀도, recall: 재현율
                              f1_score, roc_curve, auc)  # f1_score: F1 점수, roc_curve: ROC 곡선 데이터, auc: 곡선 아래 면적
import seaborn as sns                                  # seaborn: 예쁜 시각화 라이브러리 (히트맵 등)

# ── 1. 모델 학습 ─────────────────────────
data = load_breast_cancer()                            # 유방암 데이터 로드 (30개 특성, 569개 샘플)
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42)  # 80% 학습, 20% 테스트로 분할

scaler = StandardScaler()                              # 정규화 객체 생성
X_train = scaler.fit_transform(X_train)                # fit_transform: 학습 데이터에서 평균/표준편차 계산 후 정규화
X_test = scaler.transform(X_test)                      # transform: 학습 때 계산한 값으로 테스트 데이터 정규화

model = LogisticRegression(max_iter=1000)              # 로지스틱 회귀 모델 생성
model.fit(X_train, y_train)                            # fit: 학습 데이터로 모델 학습
y_pred = model.predict(X_test)                         # predict: 테스트 데이터의 클래스(0/1) 예측
y_prob = model.predict_proba(X_test)[:, 1]             # predict_proba: 양성(1)일 확률만 추출 (ROC 곡선에 사용)

# ── 2. 혼동행렬 ──────────────────────────
cm = confusion_matrix(y_test, y_pred)                  # confusion_matrix: 실제 vs 예측 결과를 2x2 표로 정리
tn, fp, fn, tp = cm.ravel()                            # ravel: 2x2 행렬을 1차원으로 펼쳐서 TN, FP, FN, TP 추출

print("[ 혼동행렬 ]")
print(f"  TP={tp} (양성을 양성으로)")                   # True Positive: 정답=양성, 예측=양성 (올바름)
print(f"  FP={fp} (음성을 양성으로 — 1종 오류)")        # False Positive: 정답=음성, 예측=양성 (잘못된 알람)
print(f"  FN={fn} (양성을 음성으로 — 2종 오류)")        # False Negative: 정답=양성, 예측=음성 (놓침)
print(f"  TN={tn} (음성을 음성으로)")                   # True Negative: 정답=음성, 예측=음성 (올바름)

# ── 3. 지표 계산 ─────────────────────────
acc = accuracy_score(y_test, y_pred)                   # accuracy: 전체 중 맞춘 비율 = (TP+TN) / 전체
prec = precision_score(y_test, y_pred)                 # precision: 양성이라고 예측한 것 중 진짜 양성 = TP / (TP+FP)
rec = recall_score(y_test, y_pred)                     # recall: 실제 양성 중 찾아낸 비율 = TP / (TP+FN)
f1 = f1_score(y_test, y_pred)                          # f1: precision과 recall의 조화평균 (둘 다 높아야 높음)

print(f"\n[ 평가지표 ]")
print(f"  Accuracy:  {acc:.4f}  (전체 중 맞춘 비율)")
print(f"  Precision: {prec:.4f}  (양성 예측 중 진짜 양성)")
print(f"  Recall:    {rec:.4f}  (진짜 양성 중 찾은 비율)")
print(f"  F1 Score:  {f1:.4f}  (Precision과 Recall의 균형)")

# 주의: 이 데이터는 target 1=benign(양성종양), 0=malignant(악성)이고,
# sklearn 기본은 1을 positive로 계산한다. "암을 놓치지 않기" 서사로 보려면 pos_label=0으로 뒤집어야 한다.
prec_mal = precision_score(y_test, y_pred, pos_label=0)  # pos_label=0: 악성을 positive로 계산
rec_mal = recall_score(y_test, y_pred, pos_label=0)      # 악성 Recall = 실제 암 환자 중 찾아낸 비율
print(f"\n[ 악성(0)을 positive로 — 암 진단 서사 ]")
print(f"  Precision(악성): {prec_mal:.4f}")
print(f"  Recall(악성):    {rec_mal:.4f}  ← '암을 놓치지 않는' 그 Recall은 이 값")

print(f"\n[ classification_report — 실무에서 이걸 쓴다 ]")
print(classification_report(y_test, y_pred, target_names=['Malignant', 'Benign']))  # 클래스별 precision/recall/f1을 표로 출력
# ↑ 아래쪽 macro avg = 클래스별 단순 평균 (소수 클래스 동등), weighted avg = 샘플 수 가중 평균 (다수 클래스에 끌려감)
# ↑ 불균형 자체를 다루는 방법(class_weight='balanced', 리샘플링)은 Chapter 9에서

# ── 4. ROC / AUC ─────────────────────────
fpr, tpr, thresholds = roc_curve(y_test, y_prob)       # roc_curve: threshold를 바꿔가며 FPR(x축), TPR(y축) 계산
roc_auc = auc(fpr, tpr)                                # auc: ROC 곡선 아래 면적 (1에 가까울수록 좋은 모델)
print(f"[ ROC AUC = {roc_auc:.4f} ]")

# ── 5. Accuracy의 함정 시연 ──────────────
print(f"\n[ Accuracy의 함정 — 불균형 데이터 ]")
y_imbalanced = np.array([0]*990 + [1]*10)              # 불균형 데이터: 1000개 중 양성이 10개(1%)뿐
y_all_negative = np.zeros(1000)                        # 전부 음성(0)이라고 예측하는 바보 모델
acc_trap = accuracy_score(y_imbalanced, y_all_negative)  # 정확도: 990개 맞춤 → 99%!
rec_trap = recall_score(y_imbalanced, y_all_negative, zero_division=0)  # 재현율: 양성 10개 중 0개 찾음 → 0%
print(f"  전부 음성으로 예측:")
print(f"  Accuracy = {acc_trap:.1%} ← 높아 보이지만")  # 정확도만 보면 좋아 보이지만...
print(f"  Recall   = {rec_trap:.1%} ← 양성을 하나도 못 찾음!")  # 실제로는 양성을 전혀 못 찾음!

# ── 6. 시각화 ─────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))        # 2행 3열 그래프 영역

# 혼동행렬 히트맵
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],  # heatmap: 혼동행렬을 색깔로 표현 (annot: 숫자 표시, fmt='d': 정수)
            xticklabels=['Malignant', 'Benign'], yticklabels=['Malignant', 'Benign'])  # 축 라벨
axes[0, 0].set_title('Confusion Matrix')               # 제목
axes[0, 0].set_xlabel('Predicted')                     # x축: 예측 라벨
axes[0, 0].set_ylabel('Actual')                        # y축: 실제 라벨

# 지표 바 차트
metrics = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1}  # 4개 평가지표
axes[0, 1].bar(metrics.keys(), metrics.values(), color='steelblue', edgecolor='black')  # 막대 그래프
axes[0, 1].set_title('Classification Metrics')         # 제목
axes[0, 1].set_ylim(0, 1)                             # y축 범위: 0~1
for i, (k, v) in enumerate(metrics.items()):           # 각 막대 위에 수치 표시
    axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)  # 막대 위에 값 텍스트

# ROC 곡선
axes[0, 2].plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {roc_auc:.3f}')  # ROC 곡선 (왼쪽 위에 가까울수록 좋음)
axes[0, 2].plot([0, 1], [0, 1], 'r--', label='Random (AUC=0.5)')  # 랜덤 분류기의 기준선 (대각선)
axes[0, 2].fill_between(fpr, tpr, alpha=0.1)           # 곡선 아래 영역 색칠 (AUC 면적)
axes[0, 2].set_title('ROC Curve')                      # 제목
axes[0, 2].set_xlabel('False Positive Rate')           # x축: 거짓 양성률 (음성을 양성으로 잘못 판단)
axes[0, 2].set_ylabel('True Positive Rate')            # y축: 진짜 양성률 (양성을 제대로 찾은 비율)
axes[0, 2].legend()                                    # 범례

# Threshold에 따른 Precision/Recall 변화
precisions = []                                        # threshold별 precision 저장
recalls = []                                           # threshold별 recall 저장
thres_range = np.arange(0.1, 1.0, 0.05)               # 0.1~0.95 범위의 threshold 후보
for t in thres_range:                                  # 각 threshold에 대해
    y_t = (y_prob >= t).astype(int)                    # 확률이 threshold 이상이면 양성(1), 아니면 음성(0)
    precisions.append(precision_score(y_test, y_t, zero_division=0))  # 해당 threshold의 precision
    recalls.append(recall_score(y_test, y_t, zero_division=0))        # 해당 threshold의 recall
axes[1, 0].plot(thres_range, precisions, 'b-', label='Precision')  # threshold vs precision 곡선
axes[1, 0].plot(thres_range, recalls, 'r-', label='Recall')        # threshold vs recall 곡선
axes[1, 0].axvline(0.5, color='gray', linestyle='--', label='default=0.5')  # 기본 threshold 0.5 기준선
axes[1, 0].set_title('Precision/Recall vs Threshold')  # 제목
axes[1, 0].set_xlabel('Threshold')                     # x축: 분류 기준값
axes[1, 0].legend()                                    # 범례

# Accuracy 함정
axes[1, 1].bar(['Accuracy\n(misleading)', 'Recall\n(truth)'],
               [acc_trap, rec_trap],
               color=['orange', 'red'], edgecolor='black')  # 정확도 vs 재현율 비교 (불균형 데이터)
axes[1, 1].set_title('Accuracy Trap (Imbalanced)')     # 제목: 정확도의 함정
axes[1, 1].set_ylim(0, 1.1)                           # y축 범위

# 예측 확률 분포
axes[1, 2].hist(y_prob[y_test == 1], bins=15, alpha=0.7, label='Positive', color='green')  # 실제 양성의 예측 확률 분포
axes[1, 2].hist(y_prob[y_test == 0], bins=15, alpha=0.7, label='Negative', color='red')    # 실제 음성의 예측 확률 분포
axes[1, 2].axvline(0.5, color='black', linestyle='--')  # 기본 threshold 기준선
axes[1, 2].set_title('Predicted Probability Distribution')  # 제목: 예측 확률 분포
axes[1, 2].legend()                                    # 범례

plt.tight_layout()                                     # 그래프 간 간격 자동 조정
plt.savefig('2-7_output.png', dpi=100)               # 이미지 파일로 저장
plt.show()                                             # 화면에 표시

# ── 정리 ──────────────────────────────────
print("\n" + "="*50)
print("핵심 정리:")
print("  혼동행렬 → TP, FP, FN, TN 한눈에")
print("  Accuracy → 불균형 데이터에서 위험")
print("  Precision → FP 줄이기 (스팸필터)")
print("  Recall → FN 줄이기 (암 진단)")
print("  F1 → 둘 다 중요할 때")
print("  AUC → 전체적인 분류 성능")
print("="*50)
