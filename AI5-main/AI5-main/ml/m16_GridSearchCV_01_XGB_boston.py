from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, LSTM
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import numpy as np
import time as t
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from xgboost import XGBClassifier, XGBRFRegressor
import xgboost as xgb
##############
# 데이터 로드 및 전처리
dataset = load_boston()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['target'] = dataset['target']
#df.boxplot()  #CRIM ZN B
#df.plot.box()

x = df.drop(['target'], axis=1).copy()
y = df['target']
print(x.shape, y.shape)

x['CRIM'] = np.log1p(x['CRIM'])
x['ZN'] = np.log1p(x['ZN'])
x['B'] = np.log1p(x['B'])


# 훈련 및 테스트 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=3)
print(x_train.shape)
print(x_test.shape)
y_train = np.log1p(y_train)
y_test = np.log1p(y_test)

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)

paramgrid = [
    {'n_jobs' : [-1], 'n_estimators' : [100, 500], 'max_depth' : [6, 10 ,12],
     'min_samples_leaf' : [3, 10], 'tree_method' : ['gpu_hist']}, #12
    {'n_jobs' : [-1], 'max_depth' : [6, 8,  10 ,12],
     'min_samples_leaf' : [3, 5, 7, 10], 'tree_method' : ['gpu_hist']}, #16
    {'n_jobs' : [-1], 'min_samples_leaf' : [3, 5, 7, 10], 
     'min_samples_leaf' : [2, 3, 5, 10], 'tree_method' : ['gpu_hist']}, #16
    {'n_jobs' : [-1], 'min_samples_leaf' : [2, 3, 5, 10], 'tree_method' : ['gpu_hist']}, #4
] #48


model = GridSearchCV(
    estimator=XGBRFRegressor(random_state=42, use_label_encoder=False, eval_metric='rmse'),
    param_grid=paramgrid,
    cv=kfold,
    scoring='r2',  # R2 스코어를 기준으로 성능 평가
    verbose=1,
    n_jobs=-1  # 병렬 처리로 속도 향상
)


# 모델 컴파일 및 훈련
model.fit(x_train, y_train )

# 평가 및 예측
y_predict = model.predict(x_test)

# R2 스코어 계산
r2 = r2_score(y_test, y_predict)
print("R2 스코어:", r2)

#R2 스코어: 0.7796704150990024

#DNN
#로스 : 0.032336339354515076
#R2 스코어: 0.7525892865336938