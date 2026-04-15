# 23_1 copy

from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
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
datasets = load_wine()      # feature_name 때문에
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
    metric_name='mlogloss', #error?
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
    eval_metric='mlogloss',             # 2.1.1 버전에서는 여기에 위치
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
[86]    validation_0-mlogloss:0.18841
최종 점수 : 0.9444444444444444
acc_score : 0.9444444444444444
[0.00028813 0.0015744  0.01177566 0.02031521 0.05159921 0.06135099
 0.06449952 0.08230198 0.094424   0.12164617 0.12662466 0.17470492
 0.18889515]
Trech=0.000, n=13, ACC: 97.22%
Trech=0.002, n=12, ACC: 97.22%
Trech=0.012, n=11, ACC: 97.22%
Trech=0.020, n=10, ACC: 97.22%
Trech=0.052, n=9, ACC: 94.44%
Trech=0.061, n=8, ACC: 97.22%
Trech=0.064, n=7, ACC: 97.22%
Trech=0.082, n=6, ACC: 94.44%
Trech=0.094, n=5, ACC: 91.67%
Trech=0.122, n=4, ACC: 91.67%
Trech=0.127, n=3, ACC: 88.89%
Trech=0.175, n=2, ACC: 86.11%
Trech=0.189, n=1, ACC: 69.44%
'''