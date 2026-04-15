from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from matplotlib import rc
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time as t
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import KFold
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,StratifiedKFold, KFold, cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score, accuracy_score

from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC, SVR
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
path = "./_data/dacon/diabets/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sample_submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)
#path = "./_data/따릉/" 이렇게 이용해서 pd구문 안을 짧게 만들 수 있음

print(train_csv.columns)
print(test_csv.columns)


x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']

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
ACC  [0.16769451 0.30051604 0.20948782] 
 평균 ACC 0.2259
[ 0.00661557  0.41644804  0.8020508   0.49057905  0.48706447  0.04603977
  0.0970685   0.64429028 -0.07738281  0.19865505  0.04384298  0.41057905
  0.29281318  0.66854506  0.48742162  0.59875338 -0.07438655  0.43916281
  0.58144804  1.0638272   0.06550591  0.09182832  0.89990988  0.05559187
  0.18481327  0.66579203  0.10022072  0.08682092  0.08721937  0.06303819
 -0.06349728 -0.0225022   0.25684369  0.52786974  0.79488267  0.40064018
 -0.06015209  0.37500645 -0.12446938  0.34035883  0.61339523  1.32771338
  0.49956524  0.7229756  -0.06144175  0.43926911  0.08007199 -0.05632525
  0.14012303 -0.05516773  0.09686301  0.09160288  1.00454657  0.25447108
  0.16418858  0.70505064  0.5261133   0.98670274  0.21217659 -0.16281819
  0.42356427  0.25156596 -0.11870707  0.43500151  0.29788489  0.31887428
 -0.08066961  0.12372778  0.18568146  0.52815259  0.6987079   0.23234415
  0.23163766  0.68740672  0.32863414  0.10390844  0.01289773 -0.06266137
  0.52473475  0.02140422  0.15035535  0.1361574   0.35746439  0.00330181
  0.21926099  0.75746985  0.10326128 -0.03690864  0.17676322  0.20258526
 -0.08324288  0.18110505  0.23363776  0.35098274  0.7785766   0.1803603
  0.09972486  0.74212711  0.67110878  0.2230444   0.7908904   0.18072122
  0.47100754  0.27390835  0.08689121  0.34255132  0.1151681   0.29082205
  0.42340524  0.32950998  0.33643851  0.04652405  0.02779226  0.04777527
  0.68848011  0.21521913  0.15317736  0.26174394  0.83366022  0.79601379
  0.43406893  0.41730524  0.68465189  0.38841339 -0.11065883  0.72266244
 -0.05099805  0.3602841   0.25102106  0.31027551  0.08045632]
ID
TRAIN_319    0
TRAIN_649    0
TRAIN_041    0
TRAIN_592    0
TRAIN_644    1
            ..
TRAIN_397    0
TRAIN_362    1
TRAIN_146    0
TRAIN_268    0
TRAIN_150    0
Name: Outcome, Length: 131, dtype: int64
cross_val_predict ACC :  0.056686484970482565
'''