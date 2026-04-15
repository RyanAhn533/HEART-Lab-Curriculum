import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score
from bayes_opt import BayesianOptimization
import time
import warnings
warnings.filterwarnings('ignore')



#1. 데이터
dataset = load_digits()
print(dataset)
print(dataset.DESCR)
print(dataset.feature_names)
random_state1 = 1223

x = dataset.data
y = dataset.target
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
    
    model = XGBClassifier(**params, n_jobs=-1)
    model.fit(x_train, y_train,
              eval_set=[(x_test, y_test)],
              #eval_metrics='logloss',
              verbose=0)
    
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
| 105       | 0.8111    | 1.0       | 0.1       | 48.78     | 10.0      | 10.0      | 36.01     | 40.0      | 50.0      | -0.001    | 1.0       |
=================================================================================================================================================
{'target': 0.9611111111111111, 'params': {'colsample_bytree': 1.0, 'learning_rate': 0.1, 'max_bin': 107.81709341124194, 'max_depth': 10.0, 'min_child_samples': 10.0, 'min_child_weight': 1.0, 'num_leaves': 24.0, 'reg_alpha': 0.01, 'reg_lambda': -0.001, 'subsample': 1.0}}
100 번 걸린 시간은?  49.85
'''