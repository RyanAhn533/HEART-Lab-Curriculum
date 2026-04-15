import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score
from bayes_opt import BayesianOptimization
import time
import warnings
warnings.filterwarnings('ignore')
import pandas as pd


#1. 데이터
path = "./_data/dacon/diabets/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sample_submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)
#path = "./_data/따릉/" 이렇게 이용해서 pd구문 안을 짧게 만들 수 있음

print(train_csv.columns)
print(test_csv.columns)


x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']
random_state = 5656

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=random_state, train_size=0.8,)

scaler = StandardScaler()
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
    
    model = XGBClassifier(**params, n_jobs=-1)
    model.fit(x_train, y_train,
              eval_set=[(x_test, y_test)],
              #eval_metrics='logloss',
              verbose=2)
    
    y_predict = model.predict(x_test)
    results = accuracy_score(y_test, y_predict)
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
| 105       | 0.7328    | 0.8707    | 0.1       | 245.6     | 7.267     | 89.68     | 31.05     | 27.79     | 4.565     | 9.753     | 0.8775    |
=================================================================================================================================================
{'target': 0.7862595419847328, 'params': {'colsample_bytree': 1.0, 'learning_rate': 0.1, 'max_bin': 50.553905910709894, 'max_depth': 3.0, 'min_child_samples': 100.0, 'min_child_weight': 1.0, 'num_leaves': 28.31276545569849, 'reg_alpha': 0.01, 'reg_lambda': 0.02075964271394122, 'subsample': 1.0}}
100 번 걸린 시간은?  22.19
'''