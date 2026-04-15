from sklearn.model_selection import GridSearchCV
from xgboost import XGBRFRegressor, XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, KFold

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
'''
parameters = [
    {'n_jobs' : [-1], 'n_estimators' : [100, 500], 'max_depth' : [6, 10 ,12],
     'min_samples_leaf' : [3, 10], 'tree_method' : ['gpu_hist']}, #12
    {'n_jobs' : [-1], 'max_depth' : [6, 8,  10 ,12],
     'min_samples_leaf' : [3, 5, 7, 10], 'tree_method' : ['gpu_hist']}, #16
    {'n_jobs' : [-1], 'min_samples_leaf' : [3, 5, 7, 10], 
     'min_samples_leaf' : [2, 3, 5, 10], 'tree_method' : ['gpu_hist']}, #16
    {'n_jobs' : [-1], 'min_samples_leaf' : [2, 3, 5, 10], 'tree_method' : ['gpu_hist']}, #4
] 
'''
parameters = [
    {'n_jobs':[-1,], 'n_estimators': [100,500], 'max_depth':[6,10,12], 'min_sample_leaf':[3,10], 'learning_rate':[0.1,0.01,0.001,0.005],'subsample': [0.8, 1.0],'tree_method': ['gpu_hist']},
    {'n_jobs':[-1,], 'max_depth':[6,8, 10,12], 'min_sample_leaf':[3,5,7,10],'subsample': [0.8, 1.0],'tree_method': ['gpu_hist']}, # 16
    {'n_jobs':[-1,], 'min_sample_leaf':[3,5,7,10], 'min_sample_split':[2,3,5,10], 'learning_rate':[0.01,0.001,0.005],'subsample': [0.8, 1.0],'tree_method': ['gpu_hist']},
    {'n_jobs':[-1,], 'min_sample_split':[2,3,5,10],'subsample': [0.8, 1.0],'tree_method': ['gpu_hist']}, 
]
import xgboost as xgb

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
'''
최적의 매개변수 :  XGBRegressor(base_score=None, booster=None, callbacks=None,
             colsample_bylevel=None, colsample_bynode=None,
             colsample_bytree=None, device=None, early_stopping_rounds=None,
             enable_categorical=False, eval_metric=None, feature_types=None,
             gamma=None, grow_policy=None, importance_type=None,
             interaction_constraints=None, learning_rate=0.1, max_bin=None,
             max_cat_threshold=None, max_cat_to_onehot=None,
             max_delta_step=None, max_depth=10, max_leaves=None,
             min_child_weight=None, min_sample_leaf=10, missing=nan,
             monotone_constraints=None, multi_strategy=None, n_estimators=500,
             n_jobs=-1, num_parallel_tree=None, ...)
최적의 파라미터 :  {'tree_method': 'gpu_hist', 'subsample': 0.8, 'n_jobs': -1, 'n_estimators': 500, 'min_sample_leaf': 10, 'max_depth': 10, 'learning_rate': 0.1}
최고의 점수 :  0.8527785623523293
모델의 점수 :  0.8768644152623404
accuracy_score :  0.8768644152623404
최적 튠 ACC:  0.8768644152623404
'''