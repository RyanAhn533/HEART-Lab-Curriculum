import sklearn as sk
from sklearn.datasets import load_digits
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time as t
import matplotlib.pyplot as plt
from matplotlib import rc
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,StratifiedKFold, KFold, cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score, accuracy_score

from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC, SVR
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import xgboost as xgb

###############

x, y = load_digits(return_X_y=True)
print(x)
print(y)
print(x.shape, y.shape)
print(pd.value_counts(y,sort=True))


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
model = xgb.XGBClassifier()

scores = cross_val_score(model, x_train, y_train, cv=kfold)
print('ACC ', scores, '\n 평균 ACC', round(np.mean(scores), 4))

y_predict = cross_val_predict(model, x_test, y_test)

print(y_predict)
print(y_test)

r2 = r2_score(y_test, y_predict)
print('cross_val_predict ACC : ', r2)

'''
[[ 0.  0.  5. ...  0.  0.  0.] 
 [ 0.  0.  0. ... 10.  0.  0.] 
 [ 0.  0.  0. ... 16.  9.  0.] 
 ...
 [ 0.  0.  1. ...  6.  0.  0.] 
 [ 0.  0.  2. ... 12.  0.  0.] 
 [ 0.  0. 10. ... 12.  1.  0.]]
[0 1 2 ... 8 9 8] 
(1797, 64) (1797,)
3    183
1    182
5    182
4    181
6    181
9    180
7    179
0    178
2    177
8    174
dtype: int64
ACC  [0.94780793 0.94989562 0.95824635] 
 평균 ACC 0.952
[5 9 9 6 1 6 6 9 7 7 4 2 1 4 5 1 4 7 0 1 4 1 3 7 0 3 7 0 4 9 1 3 2 3 1 3 1
 4 2 5 4 5 2 6 1 3 1 7 5 2 7 4 6 2 9 5 1 5 9 3 9 8 4 8 2 9 6 1 0 2 6 3 8 0
 9 1 0 3 3 9 1 7 0 0 1 3 8 8 0 6 9 5 4 0 7 3 5 7 2 9 1 8 4 3 0 3 1 9 0 5 0
 4 1 6 8 8 7 9 2 4 6 0 7 3 6 2 7 6 9 6 5 1 0 7 1 0 6 1 9 5 3 4 0 8 9 6 5 6
 2 1 2 3 3 6 1 5 7 9 8 4 7 1 0 2 7 7 7 0 8 6 0 0 7 6 9 6 0 6 4 2 4 1 3 0 3
 8 6 9 1 0 2 7 4 4 7 7 4 5 9 4 7 2 6 9 5 8 9 0 2 6 1 9 2 5 5 9 7 0 7 7 0 4
 2 3 9 9 3 1 6 0 5 6 0 2 2 6 6 3 4 3 8 3 9 9 6 1 6 5 2 4 7 9 5 0 4 5 0 7 3
 0 2 0 7 8 7 4 3 0 8 2 7 6 9 9 6 3 2 9 5 7 1 3 9 0 0 5 5 3 8 8 2 7 1 2 9 8
 4 9 4 7 0 2 9 1 4 3 1 5 1 4 9 2 2 3 9 7 6 5 2 1 5 4 6 5 5 6 6 5 8 0 2 1 8
 7 8 3 8 3 2 9 8 7 8 3 8 0 4 6 7 5 7 5 3 8 9 2 9 2 5 6]
[5 9 9 6 1 6 6 9 8 7 4 2 1 4 3 1 4 7 0 1 4 8 2 7 0 3 8 0 4 9 8 3 2 3 8 3 1
 4 2 5 4 5 2 6 1 3 1 7 6 2 7 4 6 2 9 5 1 5 5 3 9 9 4 8 2 9 6 1 0 2 6 3 3 0
 9 1 4 3 3 9 8 7 0 0 1 3 2 8 0 3 9 5 4 0 7 1 5 7 2 9 8 8 5 3 0 3 1 9 4 5 0
 4 1 6 8 8 7 9 2 4 6 0 7 2 6 2 7 5 3 6 5 1 0 7 1 0 6 4 8 5 3 4 0 8 9 6 5 6
 2 1 2 3 3 6 1 5 7 9 8 7 9 1 0 2 7 7 7 0 8 6 0 0 4 6 9 6 0 6 4 2 4 1 3 0 2
 8 6 9 1 0 2 7 4 4 4 7 4 5 9 4 7 2 6 9 5 8 9 0 8 8 1 9 2 5 5 1 7 0 7 7 0 4
 2 3 7 9 3 1 6 0 5 6 0 2 2 6 6 3 4 3 8 3 9 9 6 1 6 5 2 4 7 9 5 5 4 5 0 7 3
 0 3 0 1 8 7 4 3 0 8 1 7 6 9 9 6 3 2 9 5 7 1 3 1 0 0 5 5 3 8 8 8 7 1 2 9 8
 4 1 4 7 0 2 9 1 4 3 1 5 1 4 9 2 2 3 8 7 6 5 2 1 5 4 6 5 5 6 6 5 8 0 2 1 8
 7 8 3 8 3 2 9 8 7 5 2 8 0 4 6 7 5 7 5 9 8 9 3 4 3 5 6]
cross_val_predict ACC :  0.7068370828145747
'''