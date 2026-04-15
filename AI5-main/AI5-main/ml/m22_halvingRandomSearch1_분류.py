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
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
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


parameters = [
   #  {'learning_rate' : [0.01, 0.05, 0.1], 'max_depth' : [3 ,4 ,5],},
    {'learning_rate' : [0.01, 0.05, 0.1, 0.2, 0.3], 'max_depth' : [3 ,4 ,5, 6, 8],},
    {'learning_rate' : [0.01, 0.05, 0.1, 0.2, 0.3], 'subsample' : [0.6, 0.7, 0.8, 0.9, 1.0],},
   #  {'learning_rate' : [0.01, 0.05, 0.1, 0.2, 0.3], 'colsample_bytree' : [0.6, 0.7, 0.8, 0.9, 1.0],},
   #  {'learning_rate' : [0.01, 0.05, 0.1, 0.2, 0.3], 'gamma' : [0, 0.1, 0.2, 0.5, 1.0],},
   ] # 3*3*cv

#2. 모델 구성
model = HalvingRandomSearchCV(XGBClassifier(
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


path = 'C:\\ai5\\_save\\m17_RS_CV_01\\'
pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True) \
    .to_csv(path + 'm17_RS_CV_results.csv')

############################## 선생님의 파라미터 ###########################################
