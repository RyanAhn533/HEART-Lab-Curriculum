import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,StratifiedKFold, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import pandas as pd


#cross val score - 교차검증점수
# 5개로 짜른 데이터들마다 교차 검증 점수를 매긴다

#1. 데이터
datasets = load_iris()
df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
print(df)

n_splits=3
kfold = KFold(n_splits=n_splits, shuffle=False) #, random_state=333
#kfold 할 준비 끗

#kfold = StratifiedKFold(n_splits=n_splits, shuffle=False, random_state=333)


for train_index, val_index in kfold.split(df):
    print('=========================================')
    print('train_index', '\n', val_index)
    print('훈련데이터 개수 : ', len(train_index), " ", 
          '검증데이터 갯수', len(val_index))