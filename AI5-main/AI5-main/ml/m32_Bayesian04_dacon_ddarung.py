import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score
from bayes_opt import BayesianOptimization
import time
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

random_state1 = 777
#1. 데이터
path = "./_data/따릉이/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "submission.csv", index_col=0)
#path = "./_data/따릉/" 이렇게 이용해서 pd구문 안을 짧게 만들 수 있음

train_csv = train_csv.dropna() #train_csv 데이터에서 결측치 삭제
test_csv = test_csv.fillna(test_csv.mean()) #test_csv에는 결측치 평균으로 넣기
x = train_csv.drop(['count'], axis=1)
y = train_csv['count']
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=random_state1, train_size=0.8,
                                                    )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
bayesian_params = {
    'learning_rate' : (0.001, 0.1),
    'max_depth' : (3, 10),
    'num_leaves' : (24, 40),
    'min_child_samples' : (10, 100),
    'min_child_weight' : (1, 50),
    'subsample' : (0.5, 1),
    'colsample_bytree' : (0.5, 1),
    'max_bin' : (9, 500),
    'reg_lambda' : (-0.001, 10),
    'reg_alpha' : (0.01, 50)}

    
def xgb_hamsu(learning_rate, max_depth, num_leaves, min_child_samples,
              min_child_weight, subsample, colsample_bytree, max_bin,
             reg_lambda, reg_alpha):
    params = {'n_estimators' : 100,
              'learning_rate' : learning_rate,
              'max_depth' : int(round(max_depth,2)), 
              'num_leaves' : int(round(num_leaves)), 
              'min_child_samples' : int(round(min_child_samples)),
              'min_child_weight' : int(round(min_child_weight)), 
              'subsample' : int(round(subsample)), 
              'colsample_bytree' : int(round(colsample_bytree)), 
              'max_bin' : int(round(max_bin)),
             'reg_lambda' : int(round(reg_lambda)), 
             'reg_alpha' : int(round(reg_alpha))
              }
    
    model = XGBRegressor(**params, n_jobs=-1)
    model.fit(x_train, y_train,
              eval_set=[(x_test, y_test)],
              #eval_metrics='logloss',
              verbose=2)
    
    y_predict = model.predict(x_test)
    results = r2_score(y_test, y_predict)
    return results

#위가 블랙함수
bay = BayesianOptimization(
    f=xgb_hamsu,
    pbounds=bayesian_params,
    random_state=333,
)

n_iter = 100
start_time = time.time()
bay.maximize(init_points=5, n_iter=n_iter)
end_time = time.time()

print(bay.max)
print(n_iter, '번 걸린 시간은? ', round(end_time-start_time, 2))

'''
| 105       | 0.7474    | 0.5392    | 0.07378   | 206.4     | 3.988     | 68.98     | 13.74     | 34.17     | 26.78     | 5.434     | 0.6805    |
=================================================================================================================================================
{'target': 0.7548434159991358, 'params': {'colsample_bytree': 0.8197985649499426, 'learning_rate': 0.1, 'max_bin': 17.747425630204425, 'max_depth': 8.156791700488487, 'min_child_samples': 43.26651874361912, 'min_child_weight': 6.777454064528168, 'num_leaves': 29.608547152400465, 'reg_alpha': 6.059398993365615, 'reg_lambda': 2.8665300694650755, 'subsample': 1.0}}
100 번 걸린 시간은?  19.32
'''