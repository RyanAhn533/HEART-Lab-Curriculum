import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
#cross val score - 교차검증점수
# 5개로 짜른 데이터들마다 교차 검증 점수를 매긴다

#1. 데이터
x,y = load_iris(return_X_y=True)
print(x)

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)
#kfold 할 준비 끗

#2. model
model = SVC()
scores = cross_val_score(model, x, y, cv=kfold)
print('ACC : ', scores, '\n 평균 ACC : ', round(np.mean(scores), 4))