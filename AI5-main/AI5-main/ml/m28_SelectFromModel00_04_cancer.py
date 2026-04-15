from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
# 1. 데이터
datasets = load_breast_cancer()      # feature_name 때문에
x = datasets.data
y = datasets.target

random_state1=1223
random_state2=1223

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, stratify=y, random_state=random_state1)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

early_stop = xgb.callback.EarlyStopping(
    rounds=50,              # patience
    metric_name='logloss', #error?
    data_name='validation_0',
    save_best=True,
)

#2. 모델 구성
model = XGBClassifier(
    n_estimators = 500,
    max_depth = 6,
    gamma = 0,
    min_child_weight = 0,
    subsample = 0.4,
    reg_alpha = 0,      # L1 규제
    reg_lambda = 1,     # L1 규제
    eval_metric='logloss',             # 2.1.1 버전에서는 여기에 위치
    callbacks=[early_stop],
    random_state=3377,                  # 명시
)

#3. 훈련
model.fit(x_train, y_train, 
          eval_set=[(x_test, y_test)],
        #   eval_metric='mlogloss',     # 2.1.1 버전에서는 위에 위치
          verbose=1)

#4. 평가, 예측
result = model.score(x_test, y_test)
print('최종 점수 :', result)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print('acc_score :', acc)
print(model.feature_importances_)

thresholds = np.sort(model.feature_importances_) #sort = 오름차순
print(thresholds)

from sklearn.feature_selection import SelectFromModel

for i in thresholds:
    selection = SelectFromModel(model, threshold=i, prefit=False)

    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)

    select_model = XGBClassifier(
        n_estimators = 1000,
        max_depth = 6,
        gamma = 0,
        min_child_weight = 0,
        subsample = 0.4,
        reg_alpha = 0,      # L1 규제
        # reg_lambda = 1,     # L2 규제
        # eval_metric='logloss',            # 2.1.1 버전에서는 여기에 위치
        # callbacks=[early_stop],
        random_state=3377,                  # 명시
        )

    select_model.fit(select_x_train, y_train,
                    eval_set=[(select_x_test, y_test)], 
                    verbose=0,
                    )

    select_y_predict = select_model.predict(select_x_test)
    score = accuracy_score(y_test, select_y_predict)

    print('Trech=%.3f, n=%d, ACC: %.2f%%' %(i, select_x_train.shape[1], score*100))
    
'''
65]    validation_0-logloss:0.28202
최종 점수 : 0.9298245614035088
acc_score : 0.9298245614035088
[0.00356788 0.01899257 0.01016155 0.         0.00242976 0.03277327
 0.02098598 0.18600109 0.00642162 0.         0.         0.
 0.01755977 0.02240878 0.02884294 0.         0.         0.
 0.0140283  0.03232827 0.06956833 0.046228   0.32559824 0.06417166
 0.01928803 0.00194976 0.03508421 0.02764327 0.01396665 0.        ]
[0.         0.         0.         0.         0.         0.
 0.         0.         0.00194976 0.00242976 0.00356788 0.00642162
 0.01016155 0.01396665 0.0140283  0.01755977 0.01899257 0.01928803
 0.02098598 0.02240878 0.02764327 0.02884294 0.03232827 0.03277327
 0.03508421 0.046228   0.06417166 0.06956833 0.18600109 0.32559824]
Trech=0.000, n=30, ACC: 92.98%
Trech=0.000, n=30, ACC: 92.98%
Trech=0.000, n=30, ACC: 92.98%
Trech=0.000, n=30, ACC: 92.98%
Trech=0.000, n=30, ACC: 92.98%
Trech=0.000, n=30, ACC: 92.98%
Trech=0.000, n=30, ACC: 92.98%
Trech=0.000, n=30, ACC: 92.98%
Trech=0.002, n=22, ACC: 93.86%
Trech=0.002, n=21, ACC: 93.86%
Trech=0.004, n=20, ACC: 93.86%
Trech=0.006, n=19, ACC: 93.86%
Trech=0.010, n=18, ACC: 92.98%
Trech=0.014, n=17, ACC: 92.98%
Trech=0.014, n=16, ACC: 93.86%
Trech=0.018, n=15, ACC: 92.98%
Trech=0.019, n=14, ACC: 92.98%
Trech=0.019, n=13, ACC: 93.86%
Trech=0.021, n=12, ACC: 92.98%
Trech=0.022, n=11, ACC: 92.98%
Trech=0.028, n=10, ACC: 92.11%
Trech=0.029, n=9, ACC: 92.11%
Trech=0.032, n=8, ACC: 92.98%
Trech=0.033, n=7, ACC: 92.98%
Trech=0.035, n=6, ACC: 93.86%
Trech=0.046, n=5, ACC: 93.86%
Trech=0.064, n=4, ACC: 89.47%
Trech=0.070, n=3, ACC: 91.23%
Trech=0.186, n=2, ACC: 90.35%
Trech=0.326, n=1, ACC: 90.35%
'''