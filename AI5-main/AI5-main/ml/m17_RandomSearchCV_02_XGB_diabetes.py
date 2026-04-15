from sklearn.datasets import load_diabetes
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRFRegressor
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import xgboost as xgb
from xgboost import XGBRFRegressor, XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV

# 1. 데이터 로드
dataset = load_diabetes()
print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target

# 2. 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=123, train_size=0.8)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)

# XGBRFRegressor용 하이퍼파라미터 그리드
parameters = [
    {'n_jobs':[-1,], 'n_estimators': [100,500], 'max_depth':[6,10,12], 'min_sample_leaf':[3,10], 'learning_rate':[0.1,0.01,0.001,0.005],'subsample': [0.8, 1.0],'tree_method': ['gpu_hist']},
    {'n_jobs':[-1,], 'max_depth':[6,8, 10,12], 'min_sample_leaf':[3,5,7,10],'subsample': [0.8, 1.0],'tree_method': ['gpu_hist']}, # 16
    {'n_jobs':[-1,], 'min_sample_leaf':[3,5,7,10], 'min_sample_split':[2,3,5,10], 'learning_rate':[0.01,0.001,0.005],'subsample': [0.8, 1.0],'tree_method': ['gpu_hist']},
    {'n_jobs':[-1,], 'min_sample_split':[2,3,5,10],'subsample': [0.8, 1.0],'tree_method': ['gpu_hist']}, 
]

model = RandomizedSearchCV(xgb.XGBRegressor(),
                     parameters, cv=kfold,
                     verbose=True,
                     refit=True,
                     n_jobs=-1, 
                     n_iter=10
                     ) 
# 모델 훈련
model.fit(x_train, y_train,
          eval_set = [(x_train, y_train), (x_test, y_test)],
          verbose = True)

# 최적의 하이퍼파라미터 출력
print("최적 하이퍼파라미터:", model.best_params_)

# 평가 및 예측
y_predict = model.predict(x_test)


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

print('r2 : ', r2_score(y_test, y_predict)) 

y_pred_best = model.best_estimator_.predict(x_test) 

print('최적 튠 ACC: ', r2_score(y_test, y_pred_best)) 

'''
최적의 매개변수 :  XGBRegressor(base_score=None, booster=None, callbacks=None,
             colsample_bylevel=None, colsample_bynode=None,
             colsample_bytree=None, device=None, early_stopping_rounds=None,
             enable_categorical=False, eval_metric=None, feature_types=None,
             gamma=None, grow_policy=None, importance_type=None,
             interaction_constraints=None, learning_rate=0.01, max_bin=None,
             max_cat_threshold=None, max_cat_to_onehot=None,
             max_delta_step=None, max_depth=6, max_leaves=None,
             min_child_weight=None, min_sample_leaf=3, missing=nan,
             monotone_constraints=None, multi_strategy=None, n_estimators=500,
             n_jobs=-1, num_parallel_tree=None, ...)
최적의 파라미터 :  {'tree_method': 'gpu_hist', 'subsample': 0.8, 'n_jobs': -1, 'n_estimators': 500, 'min_sample_leaf': 3, 'max_depth': 6, 'learning_rate': 0.01}
최고의 점수 :  0.3810789160539939
모델의 점수 :  0.5313370659342695
r2 :  0.5313370659342695
최적 튠 ACC:  0.5313370659342695
'''