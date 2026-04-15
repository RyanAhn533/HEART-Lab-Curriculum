import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
from xgboost import XGBClassifier
#import lightgbm   #pip install 필요
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=4444, train_size=0.8,
                                                    stratify=y)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델
xgb = XGBClassifier()
lg = LGBMClassifier()
cat = CatBoostClassifier()
#model = XGBClassifier()

model = VotingClassifier(
    estimators=[('XGB', xgb), ('LG', lg), ('CAT', cat)],
    voting='hard', #디폴트
    #voting='soft',
)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가에측
results = model.score(x_test, y_test)
print('최종점수 :', results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('acc_score : ', acc)

'''
XGB
(tf274gpu) C:\ai5> c: && cd c:\ai5 && cmd /C "c:\Anaconda\envs\tf274gpu\python.exe c:\Users\Ryan\.vscode\extensions\ms-python.debugpy-2024.12.0-win32-x64\bundled\libs\debugpy\adapter/../..\debugpy\launcher 53760 -- c:\ai5\study\ml\m38_Voting01_cancer.py "
최종점수 : 0.9649122807017544
acc_score :  0.9649122807017544


999:    learn: 0.0098181        total: 1.71s    remaining: 0us
최종점수 : 0.956140350877193
acc_score :  0.956140350877193

'''