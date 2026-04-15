import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, accuracy_score
import xgboost as xgb
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, stratify=y, random_state=3377)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

early_stop = xgb.callback.EarlyStopping(
    rounds=50,              # patience
    metric_name='logloss', #error?
    data_name='validation_0',
    #save_best=True,
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

'''
[155]   validation_0-logloss:0.07036
최종 점수 : 0.9912280701754386
acc_score : 0.9912280701754386
'''

print(model.feature_importances_)
'''
[3.1095673e-03 2.6227774e-02 3.9051243e-03 1.6615657e-05 2.0730145e-02
 8.1776241e-03 5.2664266e-03 8.4118381e-02 7.8737978e-03 4.9598357e-03
 1.3275798e-02 1.4882818e-03 5.9160744e-03 6.1758894e-02 1.0046216e-02
 7.9790084e-03 9.9304272e-03 1.4152890e-02 1.5800500e-02 5.6784889e-03
 5.2224431e-02 4.9101923e-02 7.0977546e-02 2.1681710e-01 2.2077497e-02
 1.6301911e-02 2.9043932e-02 1.9351867e-01 1.4708905e-02 2.4816213e-02]
'''

thresholds = np.sort(model.feature_importances_) #sort = 오름차순
print(thresholds)
'''
[3.1095673e-03 2.6227774e-02 3.9051243e-03 1.6615657e-05 2.0730145e-02
 8.1776241e-03 5.2664266e-03 8.4118381e-02 7.8737978e-03 4.9598357e-03
 1.3275798e-02 1.4882818e-03 5.9160744e-03 6.1758894e-02 1.0046216e-02
 7.9790084e-03 9.9304272e-03 1.4152890e-02 1.5800500e-02 5.6784889e-03
 5.2224431e-02 4.9101923e-02 7.0977546e-02 2.1681710e-01 2.2077497e-02
 1.6301911e-02 2.9043932e-02 1.9351867e-01 1.4708905e-02 2.4816213e-02]
[1.6615657e-05 1.4882818e-03 3.1095673e-03 3.9051243e-03 4.9598357e-03
 5.2664266e-03 5.6784889e-03 5.9160744e-03 7.8737978e-03 7.9790084e-03
 8.1776241e-03 9.9304272e-03 1.0046216e-02 1.3275798e-02 1.4152890e-02
 1.4708905e-02 1.5800500e-02 1.6301911e-02 2.0730145e-02 2.2077497e-02
 2.4816213e-02 2.6227774e-02 2.9043932e-02 4.9101923e-02 5.2224431e-02
 6.1758894e-02 7.0977546e-02 8.4118381e-02 1.9351867e-01 2.1681710e-01]
'''

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