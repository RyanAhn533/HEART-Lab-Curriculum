#29_5에서 가져옴
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV
import time
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.metrics import accuracy_score, r2_score
import warnings 
warnings.filterwarnings('ignore')

#1. 데이터
dataset = load_boston()

x = dataset.data
y = dataset.target
 

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7,
                                                    random_state=6666)

parameters = [
    {'n_jobs' : [-1], 'n_estimators' : [100, 500], 'max_depth' : [6, 10 ,12],
     'min_samples_leaf' : [3, 10], 'tree_method' : ['gpu_hist']}, #12
    {'n_jobs' : [-1], 'max_depth' : [6, 8,  10 ,12],
     'min_samples_leaf' : [3, 5, 7, 10], 'tree_method' : ['gpu_hist']}, #16
    {'n_jobs' : [-1], 'min_samples_leaf' : [3, 5, 7, 10], 
     'min_samples_leaf' : [2, 3, 5, 10], 'tree_method' : ['gpu_hist']}, #16
    {'n_jobs' : [-1], 'min_samples_leaf' : [2, 3, 5, 10], 'tree_method' : ['gpu_hist']}, #4
] #48



n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=3333)



model = GridSearchCV(xgb.XGBRegressor(),
                     parameters, cv=kfold,
                     verbose=True,
                     refit=True,
                     n_jobs=-1, 
                     ) 
start_time = time.time()

end_time = time.time()

print('최적의 매개변수 : ', model.best_estimator_) 

print('최적의 파라미터 : ', model.best_params_) 

print('최고의 점수 : ', model.best_score_) 

print('모델의 점수 : ', model.score(x_test, y_test)) 

y_predict = model.predict(x_test)

print('accuracy_score : ', r2_score(y_test, y_predict)) 

y_pred_best = model.best_estimator_.predict(x_test) 

print('최적 튠 ACC: ', r2_score(y_test, y_pred_best)) 

print('걸린시간 : ', round(end_time - start_time, 2), '초') 
    
'''
최적의 매개변수 :  XGBRFRegressor(base_score=None, booster=None, callbacks=None,
               colsample_bylevel=None, colsample_bytree=None, device=None,
               early_stopping_rounds=None, enable_categorical=False,
               eval_metric=None, feature_types=None, gamma=None,
               grow_policy=None, importance_type=None,
               interaction_constraints=None, max_bin=None,
               max_cat_threshold=None, max_cat_to_onehot=None,
               max_delta_step=None, max_depth=12, max_leaves=None,
               min_child_weight=None, min_samples_leaf=3, missing=nan,
               monotone_constraints=None, multi_strategy=None, n_estimators=500,
               n_jobs=-1, num_parallel_tree=None, objective='reg:squarederror',
               random_state=None, ...)
최적의 파라미터 :  {'max_depth': 12, 'min_samples_leaf': 3, 'n_estimators': 500, 'n_jobs': -1, 'tree_method': 'gpu_hist'}
최고의 점수 :  0.8373213173462846
모델의 점수 :  0.8722303582329111
accuracy_score :  0.8722303582329111
최적 튠 ACC:  0.8722303582329111
'''