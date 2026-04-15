import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,StratifiedKFold, KFold, cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score, accuracy_score

from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC, SVR
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#cross val score - 교차검증점수
# 5개로 짜른 데이터들마다 교차 검증 점수를 매긴다

#1. 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True,
                                                    random_state=123, train_size=0.8,
                                                    stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

n_splits=3

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333) #, random_state=333
#kfold 할 준비 끗

#kfold = StratifiedKFold(n_splits=n_splits, shuffle=False, random_state=333)
model = SVR()

scores = cross_val_score(model, x_train, y_train, cv=kfold)
print('ACC ', scores, '\n 평균 ACC', round(np.mean(scores), 4))

y_predict = cross_val_predict(model, x_test, y_test)

print(y_predict)
print(y_test)

r2 = r2_score(y_test, y_predict)
print('cross_val_predict ACC : ', r2)

'''
ACC  [0.91314301 0.9618145  0.92980003] 
 평균 ACC 0.9349
[ 9.13128767e-01  1.68915088e-01  2.01950103e+00  1.72149899e+00
  1.99840631e-01  6.28155373e-02  1.89362848e+00  1.31647174e+00
  1.64847801e+00  2.28221019e-02 -1.95933092e-03  1.51901537e+00
  1.96055718e+00  1.38570226e+00  1.23135650e+00  1.22178883e+00
  6.50682409e-02  8.69626192e-02 -2.31274344e-02  4.54652389e-01
 -2.43725467e-02  1.65751593e+00  1.62385295e+00  1.28808177e+00
  1.88450741e+00  1.50161404e+00  8.65309097e-01  1.31279920e+00
  9.58118745e-01  1.13564315e+00]
[1 0 2 2 0 0 2 1 2 0 0 1 2 1 2 1 0 0 0 0 0 2 2 1 2 2 1 1 1 1]   
cross_val_predict ACC :  0.8793779948255588
'''