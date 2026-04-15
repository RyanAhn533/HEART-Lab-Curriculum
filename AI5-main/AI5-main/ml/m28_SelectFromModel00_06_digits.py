# 23_1 copy

from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb

# 1. 데이터
# 1. 데이터
datasets = load_digits()      # feature_name 때문에
x = datasets.data
y = datasets.target

random_state1=1223
random_state2=1223

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, stratify=y, random_state=random_state1)

# 2. 모델 구성
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
[499]   validation_0-mlogloss:0.06158
최종 점수 : 0.9833333333333333
acc_score : 0.9833333333333333
[0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 3.8987541e-06
 2.2862103e-05 2.3235840e-05 4.3871521e-05 7.7576457e-05 2.5913320e-04
 6.3679216e-04 1.4236136e-03 1.7679266e-03 3.0631965e-03 3.3150683e-03
 3.5011757e-03 3.6015790e-03 4.3220622e-03 4.3513798e-03 4.5717289e-03
 4.9033957e-03 5.0908891e-03 5.1564225e-03 6.4571453e-03 7.3775658e-03
 8.3649196e-03 8.9949742e-03 9.1397762e-03 9.4246753e-03 9.6857734e-03
 9.8064728e-03 1.0476610e-02 1.0544425e-02 1.1742567e-02 1.3942168e-02
 1.4110732e-02 1.4775920e-02 1.4836540e-02 1.5714426e-02 1.6203623e-02
 1.6617153e-02 1.6725931e-02 1.7220087e-02 1.9950110e-02 2.0655470e-02
 2.4120357e-02 2.4564713e-02 2.5393333e-02 2.6795905e-02 2.6818149e-02
 2.7484391e-02 3.1333711e-02 3.1503621e-02 3.3001740e-02 3.3174388e-02
 3.3497926e-02 3.4185536e-02 3.5976987e-02 3.7856240e-02 3.8939845e-02
 4.8600268e-02 5.1803421e-02 5.3000562e-02 5.3046022e-02]
Trech=0.000, n=64, ACC: 98.33%
Trech=0.000, n=64, ACC: 98.33%
Trech=0.000, n=64, ACC: 98.33%
Trech=0.000, n=64, ACC: 98.33%
Trech=0.000, n=60, ACC: 98.33%
Trech=0.000, n=59, ACC: 98.33%
Trech=0.000, n=58, ACC: 98.33%
Trech=0.000, n=57, ACC: 98.33%
Trech=0.000, n=56, ACC: 98.33%
Trech=0.000, n=55, ACC: 98.33%
Trech=0.001, n=54, ACC: 98.33%
Trech=0.001, n=53, ACC: 98.33%
Trech=0.002, n=52, ACC: 98.06%
Trech=0.003, n=51, ACC: 98.06%
Trech=0.003, n=50, ACC: 97.78%
Trech=0.004, n=49, ACC: 97.50%
Trech=0.004, n=48, ACC: 97.78%
Trech=0.004, n=47, ACC: 97.78%
Trech=0.004, n=46, ACC: 98.06%
Trech=0.005, n=45, ACC: 98.06%
Trech=0.005, n=44, ACC: 98.06%
Trech=0.005, n=43, ACC: 97.78%
Trech=0.005, n=42, ACC: 98.33%
Trech=0.006, n=41, ACC: 97.78%
Trech=0.007, n=40, ACC: 98.06%
Trech=0.008, n=39, ACC: 98.33%
Trech=0.009, n=38, ACC: 98.06%
Trech=0.009, n=37, ACC: 97.50%
Trech=0.009, n=36, ACC: 97.50%
Trech=0.010, n=35, ACC: 97.22%
Trech=0.010, n=34, ACC: 97.22%
Trech=0.010, n=33, ACC: 96.39%
Trech=0.011, n=32, ACC: 96.94%
Trech=0.012, n=31, ACC: 97.22%
Trech=0.014, n=30, ACC: 96.67%
Trech=0.014, n=29, ACC: 96.11%
Trech=0.015, n=28, ACC: 96.11%
Trech=0.015, n=27, ACC: 96.67%
Trech=0.016, n=26, ACC: 96.94%
Trech=0.016, n=25, ACC: 96.67%
Trech=0.017, n=24, ACC: 96.67%
Trech=0.017, n=23, ACC: 97.78%
Trech=0.017, n=22, ACC: 97.50%
Trech=0.020, n=21, ACC: 96.67%
Trech=0.021, n=20, ACC: 96.39%
Trech=0.024, n=19, ACC: 96.39%
Trech=0.025, n=18, ACC: 96.94%
Trech=0.025, n=17, ACC: 95.83%
Trech=0.027, n=16, ACC: 95.56%
Trech=0.027, n=15, ACC: 95.83%
Trech=0.027, n=14, ACC: 95.56%
Trech=0.031, n=13, ACC: 95.28%
Trech=0.032, n=12, ACC: 93.33%
Trech=0.033, n=11, ACC: 90.00%
Trech=0.033, n=10, ACC: 87.50%
Trech=0.033, n=9, ACC: 86.67%
Trech=0.034, n=8, ACC: 83.89%
Trech=0.036, n=7, ACC: 84.17%
Trech=0.038, n=6, ACC: 77.22%
Trech=0.039, n=5, ACC: 69.17%
Trech=0.049, n=4, ACC: 54.44%
Trech=0.052, n=3, ACC: 46.94%
Trech=0.053, n=2, ACC: 31.67%
Trech=0.053, n=1, ACC: 26.39%
'''