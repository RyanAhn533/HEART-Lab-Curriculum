from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRFRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# 데이터 로드 및 전처리
dataset = load_boston()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['target'] = dataset['target']

x = df.drop(['target'], axis=1).copy()
y = df['target']

x['CRIM'] = np.log1p(x['CRIM'])
x['ZN'] = np.log1p(x['ZN'])
x['B'] = np.log1p(x['B'])

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=3)
y_train = np.log1p(y_train)
y_test = np.log1p(y_test)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)

# XGBRFRegressor용 하이퍼파라미터 그리드
parameters = [
    {
        'n_jobs': [-1],
        'n_estimators': [100, 500],          # 트리 개수
        'max_depth': [6, 10, 12],            # 트리의 최대 깊이
        'learning_rate': [0.01, 0.05, 0.1],  # 학습률 추가
        'min_child_weight': [1, 3, 5],       # 추가 파라미터: 최소 자식 노드 가중치
        'subsample': [0.8, 1.0],             # 추가 파라미터: 데이터 샘플링 비율
        'tree_method': ['gpu_hist']          # GPU 가속
    }
]

model = RandomizedSearchCV(
    estimator=XGBRFRegressor(),
    param_distributions=parameters,  # param_distributions로 변경 (RandomizedSearchCV에서는 이렇게 사용)
    cv=kfold,
    scoring='r2',  # R2 스코어를 기준으로 성능 평가
    verbose=1,
    n_jobs=-1,  # 병렬 처리로 속도 향상
    n_iter=10,# RandomizedSearchCV에서 시도할 조합의 수 (조절 가능)
    random_state=333
)

# 모델 훈련
model.fit(x_train, y_train)

# 최적의 하이퍼파라미터 출력
print("최적 하이퍼파라미터:", model.best_params_)

# 평가 및 예측
y_predict = model.predict(x_test)

# 로그 변환을 되돌려서 원래 스케일로 변환
y_test_original = np.expm1(y_test)
y_predict_original = np.expm1(y_predict)

# R2 스코어 계산
print('최적의 매개변수 : ', model.best_estimator_) 
print('최적의 파라미터 : ', model.best_params_) 
print('최고의 점수 : ', model.best_score_) 
print('모델의 점수 : ', model.score(x_test, y_test)) 
print('accuracy_score : ', r2_score(y_test, y_predict)) 

y_pred_best = model.best_estimator_.predict(x_test) 
print('최적 튠 ACC: ', r2_score(y_test, y_pred_best)) 
