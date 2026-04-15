from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRFRegressor
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
# 데이터 로드 및 전처리
dataset = fetch_california_housing()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['target'] = dataset['target']

x = df.drop(['target'], axis=1).copy()
y = df['target']
print(x.shape, y.shape)

# 데이터 스케일링
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# 훈련 및 테스트 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, train_size=0.8, shuffle=True, random_state=3)
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)

# XGBRFRegressor용 하이퍼파라미터 그리드
parameters = [
    {'n_jobs' : [-1], 'n_estimators' : [100, 500], 'max_depth' : [6, 10 ,12],
     'min_samples_leaf' : [3, 10], 'tree_method' : ['gpu_hist']}, #12
    {'n_jobs' : [-1], 'max_depth' : [6, 8,  10 ,12],
     'min_samples_leaf' : [3, 5, 7, 10], 'tree_method' : ['gpu_hist']}, #16
    {'n_jobs' : [-1], 'min_samples_leaf' : [3, 5, 7, 10], 
     'min_samples_leaf' : [2, 3, 5, 10], 'tree_method' : ['gpu_hist']}, #16
    {'n_jobs' : [-1], 'min_samples_leaf' : [2, 3, 5, 10], 'tree_method' : ['gpu_hist']}, #4
] 

model = GridSearchCV(
    estimator=XGBRFRegressor(),
    param_grid=parameters,
    cv=kfold,
    scoring='r2',  # R2 스코어를 기준으로 성능 평가
    verbose=1,
    n_jobs=-1  # 병렬 처리로 속도 향상
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
#r2 = r2_score(y_test_original, y_predict_original)
#print("R2 스코어:", r2)

# 최적 하이퍼파라미터: {'colsample_bytree': 1.0, 'learning_rate': 0.2, 'max_depth': 8, 'n_estimators': 200, 'subsample': 1.0}
# R2 스코어: 0.23218025569189593

print('최적의 매개변수 : ', model.best_estimator_) 

print('최적의 파라미터 : ', model.best_params_) 

print('최고의 점수 : ', model.best_score_) 

print('모델의 점수 : ', model.score(x_test, y_test)) 

y_predict = model.predict(x_test)

print('accuracy_score : ', r2_score(y_test, y_predict)) 

y_pred_best = model.best_estimator_.predict(x_test) 

print('최적 튠 ACC: ', r2_score(y_test, y_pred_best)) 
