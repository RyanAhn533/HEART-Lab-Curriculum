# 17_1 copy

import numpy as np
import time
import xgboost as xgb
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

'''
ImportError: HalvingGridSearchCV is experimental and the API might change without any deprecation cycle. To use it, you need to explicitly import enable_halving_search_cv:
from sklearn.experimental import enable_halving_search_cv
-> HalvingGridSearchCV는 실험용으로 enable 받아야한다 -> enable_halving_search_cv을 위에 위치
'''

#1. 데이터
x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, stratify=y, random_state=3333)

print(x_train.shape, y_train.shape)    # (1437, 64) (1437,)
print(x_test.shape, y_test.shape)      # (360, 64) (360,)

n_splits=5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=3333)

# parameters = [
#     {'learning_rate' : [0.01, 0.05, 0.1, 0.2, 0.3],
#      'max_depth' : [3 ,4 ,5, 6, 8],
#      'subsample' : [0.6, 0.7, 0.8, 0.9, 1.0],
#      'colsample_bytree' : [0.6, 0.7, 0.8, 0.9, 1.0],
#      'gamma' : [0, 0.1, 0.2, 0.5, 1.0],
#     }
#    ] # 5*5*5*5*5

parameters = [
   #  {'learning_rate' : [0.01, 0.05, 0.1], 'max_depth' : [3 ,4 ,5],},
    {'learning_rate' : [0.01, 0.05, 0.1, 0.2, 0.3], 'max_depth' : [3 ,4 ,5, 6, 8],},
    {'learning_rate' : [0.01, 0.05, 0.1, 0.2, 0.3], 'subsample' : [0.6, 0.7, 0.8, 0.9, 1.0],},
   #  {'learning_rate' : [0.01, 0.05, 0.1, 0.2, 0.3], 'colsample_bytree' : [0.6, 0.7, 0.8, 0.9, 1.0],},
   #  {'learning_rate' : [0.01, 0.05, 0.1, 0.2, 0.3], 'gamma' : [0, 0.1, 0.2, 0.5, 1.0],},
   ] # 3*3*cv

#2. 모델 구성
model = HalvingGridSearchCV(XGBClassifier(
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
                            factor=3,
                            min_resources=30, #자를 때 너무 작게 자르면 라벨이 전부 다 들어 있지 않을 수 있음. 충분히 크게 잘라줘야함
                            max_resources=1437,
                            aggressive_elimination=True, #파라미터 나눠주는 걸 팩터보다 더 큰 수로 나누어주겠따
                            )

start = time.time()
model.fit(x_train, y_train,
          eval_set=[(x_train, y_train), (x_test, y_test)],
          verbose=False
         )
end = time.time()

# factor=2 / 3*3*cv
'''
n_iterations: 4
n_required_iterations: 4
n_possible_iterations: 4
min_resources_: 179
max_resources_: 1437
aggressive_elimination: False
factor: 2
----------
iter: 0
n_candidates: 9
n_resources: 179
Fitting 5 folds for each of 9 candidates, totalling 45 fits
----------
iter: 1
n_candidates: 5
n_resources: 358
Fitting 5 folds for each of 5 candidates, totalling 25 fits
----------
iter: 2
n_candidates: 3
n_resources: 716
Fitting 5 folds for each of 3 candidates, totalling 15 fits
----------
iter: 3
n_candidates: 2
n_resources: 1432
Fitting 5 folds for each of 2 candidates, totalling 10 fits

최적의 파라미터 : {'learning_rate': 0.1, 'max_depth': 4}
best_score : 0.9433566433566434
model.score : 0.9638888888888889
accuracy_score : 0.9638888888888889
최적의 튠 ACC : 0.9638888888888889
time : 86.6 초
'''
# factor=3 / 5*5*cv
'''
n_iterations: 3
n_required_iterations: 3
n_possible_iterations: 3
min_resources_: 159
max_resources_: 1437
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 25
n_resources: 159
Fitting 5 folds for each of 25 candidates, totalling 125 fits
----------
iter: 1
n_candidates: 9
n_resources: 477
Fitting 5 folds for each of 9 candidates, totalling 45 fits
----------
iter: 2
n_candidates: 3
n_resources: 1431
Fitting 5 folds for each of 3 candidates, totalling 15 fits

최적의 파라미터 : {'learning_rate': 0.3, 'max_depth': 4}
best_score : 0.9558581769108084
model.score : 0.9694444444444444
accuracy_score : 0.9694444444444444
최적의 튠 ACC : 0.9694444444444444
time : 142.89 초
'''
# factor=3 / 5*5*cv / min_resources=100
'''
n_iterations: 3
n_required_iterations: 3
n_possible_iterations: 3
min_resources_: 100
max_resources_: 1437
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 25
n_resources: 100
Fitting 5 folds for each of 25 candidates, totalling 125 fits
----------
iter: 1
n_candidates: 9
n_resources: 300
Fitting 5 folds for each of 9 candidates, totalling 45 fits
----------
iter: 2
n_candidates: 3
n_resources: 900
Fitting 5 folds for each of 3 candidates, totalling 15 fits

최적의 파라미터 : {'learning_rate': 0.3, 'max_depth': 6}
best_score : 0.9397827436374921
model.score : 0.9638888888888889
accuracy_score : 0.9638888888888889
최적의 튠 ACC : 0.9638888888888889
time : 162.42 초
'''

print('최적의 매개변수 :', model.best_estimator_)
print('최적의 파라미터 :', model.best_params_)

print('best_score :', model.best_score_)
print('model.score :', model.score(x_test, y_test))

y_predict = model.predict(x_test)
print('accuracy_score :', accuracy_score(y_test, y_predict))    # 이전과 차이를 보기위해

y_predict_best = model.best_estimator_.predict(x_test)
print('최적의 튠 ACC :', accuracy_score(y_test, y_predict_best))

print('time :', round(end - start, 2), '초')

import pandas as pd
# print(pd.DataFrame(model.cv_results_))
'''
   mean_fit_time  std_fit_time  mean_score_time  std_score_time param_kernel  param_gamma  ...  split2_test_score  split3_test_score split4_test_score  mean_test_score  std_test_score  rank_test_score
0       0.002392  4.883640e-04         0.001196    3.981831e-04          rbf       0.0001  ...           0.916667           0.833333          0.875000         0.891667        0.062361                7
1       0.002392  4.886561e-04         0.001595    4.885582e-04      sigmoid          NaN  ...           0.000000           0.041667          0.000000         0.025000        0.020412               10
2       0.001993  9.655217e-07         0.000996    6.910027e-07       linear          NaN  ...           1.000000           0.916667          0.958333         0.966667        0.031180                1
...
7       0.001594  4.884610e-04         0.000598    4.883832e-04          rbf       0.0001  ...           0.916667           0.833333          0.916667         0.908333        0.055277                4
8       0.001395  4.880717e-04         0.000598    4.882274e-04       linear          NaN  ...           1.000000           0.875000          0.916667         0.933333        0.056519                2
9       0.001395  4.882079e-04         0.000797    3.986120e-04      sigmoid       0.0001  ...           0.916667           0.833333          0.875000         0.891667        0.062361                7
[10 rows x 17 columns]
'''
# print(pd.DataFrame(model.cv_results_).T)
'''
                                                            0                                             1  ...                                           8                                                  9
mean_fit_time                                        0.001795                                      0.002392  ...                                    0.001395                                           0.001794
std_fit_time                                         0.000398                                      0.000488  ...                                    0.000489                                           0.000399
mean_score_time                                      0.001395                                      0.000798  ...                                    0.000399                                           0.000797
std_score_time                                       0.000488                                      0.000399  ...                                    0.000488                                           0.000399
param_kernel                                              rbf                                       sigmoid  ...                                      linear                                            sigmoid
param_gamma                                            0.0001                                           NaN  ...                                         NaN                                             0.0001
param_C                                                     1                                           100  ...                                          10                                                  1
param_degree                                              NaN                                           5.0  ...                                         3.0                                                4.0
params             {'kernel': 'rbf', 'gamma': 0.0001, 'C': 1}  {'kernel': 'sigmoid', 'degree': 5, 'C': 100}  ...  {'kernel': 'linear', 'degree': 3, 'C': 10}  {'kernel': 'sigmoid', 'gamma': 0.0001, 'degree...
split0_test_score                                    0.833333                                      0.041667  ...                                         1.0                                           0.833333
split1_test_score                                         1.0                                      0.041667  ...                                       0.875                                                1.0
split2_test_score                                    0.916667                                           0.0  ...                                         1.0                                           0.916667
split3_test_score                                    0.833333                                      0.041667  ...                                       0.875                                           0.833333
split4_test_score                                       0.875                                           0.0  ...                                    0.916667                                              0.875
mean_test_score                                      0.891667                                         0.025  ...                                    0.933333                                           0.891667
std_test_score                                       0.062361                                      0.020412  ...                                    0.056519                                           0.062361
rank_test_score                                             7                                            10  ...                                           2                                                  7
[17 rows x 10 columns]
'''
# print(pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True))
'''
   mean_fit_time  std_fit_time  mean_score_time  std_score_time param_kernel  param_gamma  ...  split2_test_score  split3_test_score split4_test_score  mean_test_score  std_test_score  rank_test_score
2       0.001595      0.000488         0.000996    6.298463e-04       linear          NaN  ...           1.000000           0.916667          0.958333         0.966667        0.031180                1
5       0.001196      0.000399         0.000997    5.352484e-07       linear          NaN  ...           1.000000           0.875000          0.916667         0.933333        0.056519                2
8       0.001197      0.000399         0.000598    4.880326e-04       linear          NaN  ...           1.000000           0.875000          0.916667         0.933333        0.056519                2
...
3       0.001196      0.000399         0.000598    4.883052e-04      sigmoid       0.0001  ...           0.916667           0.833333          0.875000         0.891667        0.062361                7
9       0.001395      0.000488         0.000797    3.984929e-04      sigmoid       0.0001  ...           0.916667           0.833333          0.875000         0.891667        0.062361                7
1       0.002592      0.000489         0.000997    4.862804e-07      sigmoid          NaN  ...           0.000000           0.041667          0.000000         0.025000        0.020412               10
[10 rows x 17 columns]
'''
# print(pd.DataFrame(model.cv_results_).columns)
'''
Index(['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time',
       'param_kernel', 'param_gamma', 'param_C', 'param_degree', 'params',
       'split0_test_score', 'split1_test_score', 'split2_test_score',
       'split3_test_score', 'split4_test_score', 'mean_test_score',
       'std_test_score', 'rank_test_score'],
'''

path = 'C:\\ai5\\_save\\m17_RS_CV_01\\'
pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True) \
    .to_csv(path + 'm17_RS_CV_results.csv')

############################## 선생님의 파라미터 ###########################################
'''
parameters = [
    {
    #  'learning_rate' : [0.01, 0.05, 0.1, 0.2, 0.3],
    #  'max_depth' : [3, 4, 5, 6, 8],
    #  'subsample' : [0.6, 0.7, 0.8, 0.9, 1.0],
    #  'colsample_bytree' : [0.6, 0.7, 0.8, 0.9, 1.0],
    #  'gamma' : [0, 0.1, 0.2, 0.5, 1.0],

     'learning_rate' : [0.05],
     'max_depth' : [8],
     'subsample' : [1.0],
     'colsample_bytree' : [0.6],
     'gamma' : [0.5],

     'min_child_weight' : [1, 3, 5, 10, 15],
     'lambda' : [0, 0.1, 1, 10, 100],
     'alpha' : [0, 0.1, 1, 10, 100],
    #  'scale_pos_weight' : [1, 10, 20, 30, 40]     # version 바뀌면서 없어짐
    }
   ]

'''