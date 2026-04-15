from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
# 1. 데이터
datasets = load_diabetes()      # feature_name 때문에
x = datasets.data
y = datasets.target

random_state1=1223
random_state2=1223

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=random_state1)

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
model = XGBRegressor(
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
r2 = r2_score(y_test, y_predict)
print('r2 :', r2)

print(model.feature_importances_)
thresholds = np.sort(model.feature_importances_) #sort = 오름차순
print(thresholds)

from sklearn.feature_selection import SelectFromModel

for i in thresholds:
    selection = SelectFromModel(model, threshold=i, prefit=False)

    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)

    select_model = XGBRegressor(
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
    score = r2_score(y_test, select_y_predict)

    print('Trech=%.3f, n=%d, ACC: %.2f%%' %(i, select_x_train.shape[1], score*100))
    
'''
[51]    validation_0-logloss:-5786.01132
최종 점수 : 0.3129414436866167
r2 : 0.3129414436866167
[0.02109652 0.08766713 0.26204118 0.07920834 0.0636018  0.03776088
 0.04487437 0.0026171  0.3190009  0.08213176]
[0.0026171  0.02109652 0.03776088 0.04487437 0.0636018  0.07920834
 0.08213176 0.08766713 0.26204118 0.3190009 ]
Trech=0.003, n=10, ACC: 23.17%
Trech=0.021, n=9, ACC: 23.74%
Trech=0.038, n=8, ACC: 9.22%
Trech=0.045, n=7, ACC: 16.77%
Trech=0.064, n=6, ACC: 23.47%
Trech=0.079, n=5, ACC: 16.48%
Trech=0.082, n=4, ACC: 9.20%
Trech=0.088, n=3, ACC: 0.64%
Trech=0.262, n=2, ACC: 10.91%
Trech=0.319, n=1, ACC: -38.25%
'''