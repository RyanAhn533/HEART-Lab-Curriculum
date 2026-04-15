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
#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

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
ACC  [0.79249369 0.89460715 0.82294849] 
 평균 ACC 0.8367
[ 0.36823172 -0.05619661  1.03173679  0.8731846   0.89568227 -0.06911139
  0.84860645  0.1506989   0.91702297  1.01909623  1.17499315  1.11977766
  1.07165689  0.92734056  0.68808733  0.17953924  1.10076229  0.24804021
  0.94961396  0.50978787  0.97747816  0.56009295  0.26278833  0.68159454
  0.04717949  1.0797043   0.99566034  1.08181437  0.86577596  0.2434102
  1.19698053  0.8513485  -0.21466552  0.94342976  0.82222406 -0.01334422
  1.01385462  1.0141987   0.68358048 -0.06696116  0.83745905  0.23985369
  0.78329766  0.70604647  0.18405632  0.96883928  0.98223213  0.03837719
  1.08277875  1.00552606  0.04674513  0.99267635  0.93261763  1.0508924
  1.04278172 -0.11786698  0.24130237 -0.14444331  0.08270286  0.40860314
  1.0144591   1.03360892  0.80791144  0.11921124 -0.00587558  0.39885118
  0.15601671  0.9889178   0.9399332   0.03839768  0.62754196  0.12179116
  0.20132296  0.85304346  1.14914048  0.19876228  0.95074745  0.07788126
  0.36131144  0.14295045  0.77568948 -0.07481764  1.04664783  1.13717616
  0.18833275  0.73893004 -0.01867394  1.03601052  1.0120656  -0.01298436
  0.26062484  0.56418451  1.10072384  0.34450661  0.53397837  0.92449569
 -0.02084995  1.10995888  0.85472914  0.07327544  1.06541006  1.05243876
  1.07840161  0.66304028  1.02120816  0.88803618  1.07704144  0.19582951
  0.43284442  1.08155373  1.09291092  0.81670088  1.17141259  0.98153317]
[0 0 1 1 1 0 1 0 1 1 1 1 1 1 1 0 1 1 1 1 1 1 0 1 0 1 1 1 1 0 1 1 0 1 1 0 1
 1 0 0 1 1 1 1 0 1 1 0 1 1 0 1 1 1 1 0 0 0 0 0 1 1 1 0 0 1 0 1 1 0 1 0 0 1
 1 0 1 0 0 0 1 0 1 1 0 1 0 1 1 0 0 1 1 0 0 1 0 1 1 0 1 1 1 0 1 1 1 0 0 1 1
 1 1 1]
cross_val_predict ACC :  0.7770764234452294
'''