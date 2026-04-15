import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
import pandas as pd


#1.데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, 
                                                    random_state=333,
                                                    stratify=y, train_size=0.8)

n_splits=5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=333)

parametes = [{"C":[1, 10, 100, 500], 'kernel':['linear', 'sigmoid'],
              'degree':[3,4,5]}, #24
             {'C':[1, 10, 100], 'kernel':['rbf'], 'gamma':[0.001, 0.0001]}, #6,
             {'C':[1, 10, 100, 1000], 'kernel':['sigmoid'], 'gamma':[0.01, 0.001, 0.0001], 'degree':[3,4]} #24
             ] #총 54

#model = XGboost