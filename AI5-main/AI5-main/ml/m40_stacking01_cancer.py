import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
from sklearn.ensemble import VotingClassifier, StackingClassifier, StackingRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
rf = RandomForestClassifier()
model = StackingClassifier(estimators=[('XGB', xgb), ('RF', rf), ('CAT', cat)],
                           final_estimator=CatBoostClassifier(verbose=0),
                           n_jobs=-1,
                           cv=5)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_pred = model.predict(x_test)
print('model.score : ', model.score(x_test, y_test))
print('스태킹 ACC', accuracy_score(y_test, y_pred))