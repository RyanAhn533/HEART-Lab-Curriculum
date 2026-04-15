# [실습] 만들기

import numpy as np
import time
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score

'''
ImportError: HalvingGridSearchCV is experimental and the API might change without any deprecation cycle. To use it, you need to explicitly import enable_halving_search_cv:
from sklearn.experimental import enable_halving_search_cv
-> HalvingGridSearchCV는 실험용으로 enable 받아야한다 -> enable_halving_search_cv을 위에 위치
'''

#1. 데이터
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=3333)

print(x_train.shape, y_train.shape)    # (353, 10) (353,)
print(x_test.shape, y_test.shape)      # (89, 10) (89,)

scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)

parameters = [
   #  {'learning_rate' : [0.01, 0.05, 0.1], 'max_depth' : [3 ,4 ,5],},
    {'learning_rate' : [0.01, 0.05, 0.1, 0.2, 0.3], 'max_depth' : [3 ,4 ,5, 6, 8],},
    {'learning_rate' : [0.01, 0.05, 0.1, 0.2, 0.3], 'subsample' : [0.6, 0.7, 0.8, 0.9, 1.0],},
    {'learning_rate' : [0.01, 0.05, 0.1, 0.2, 0.3], 'colsample_bytree' : [0.6, 0.7, 0.8, 0.9, 1.0],},
    {'learning_rate' : [0.01, 0.05, 0.1, 0.2, 0.3], 'gamma' : [0, 0.1, 0.2, 0.5, 1.0],},
   ] # 3*3*cv

#2. 모델 구성
model = HalvingRandomSearchCV(XGBRegressor(
                                            # tree_mothod='gpu_hist',
                                            tree_mothod='hist',
                                            device='cuda',
                                            n_estimators=50,
                                            ),
                            parameters,
                            cv=kfold,
                            verbose=1,      # 1:이터레이터만 출력, 2이상:훈련도 출력
                            refit=True,
                            #  n_jobs=-1,
                            #  n_iter=10,
                            random_state=333,
                            factor=2,       # Default: 3, float도 가능
                           #  min_resources=100,
                           #  max_resources=1437,
                            aggressive_elimination=False,
                            min_resources=30
                            )

start = time.time()
model.fit(x_train, y_train,
          eval_set=[(x_train, y_train), (x_test, y_test)],
          verbose=False
         )
end = time.time()



print('최적의 매개변수 :', model.best_estimator_)
print('최적의 파라미터 :', model.best_params_)

print('best_score :', model.best_score_)
print('model.score :', model.score(x_test, y_test))

y_predict = model.predict(x_test)
print('accuracy_score :', r2_score(y_test, y_predict))    # 이전과 차이를 보기위해

y_predict_best = model.best_estimator_.predict(x_test)
print('최적의 튠 ACC :', r2_score(y_test, y_predict_best))

print('time :', round(end - start, 2), '초')

import pandas as pd
path = 'C:\\프로그램\\ai5\\_save\\M21\\'
pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True) \
    .to_csv(path + 'm22.csv')

'''
최적의 매개변수 : XGBRegressor(base_score=None, booster=None, callbacks=None,
             colsample_bylevel=None, colsample_bynode=None,
             colsample_bytree=0.6, device='cuda', early_stopping_rounds=None,
             enable_categorical=False, eval_metric=None, feature_types=None,
             gamma=None, grow_policy=None, importance_type=None,
             interaction_constraints=None, learning_rate=0.1, max_bin=None,
             max_cat_threshold=None, max_cat_to_onehot=None,
             max_delta_step=None, max_depth=None, max_leaves=None,
             min_child_weight=None, missing=nan, monotone_constraints=None,
             multi_strategy=None, n_estimators=50, n_jobs=None,
             num_parallel_tree=None, random_state=None, ...)
최적의 파라미터 : {'learning_rate': 0.1, 'colsample_bytree': 0.6}
best_score : 0.3494900217855653
model.score : 0.350105217994254
accuracy_score : 0.350105217994254
최적의 튠 ACC : 0.350105217994254
time : 23.91 초

'''