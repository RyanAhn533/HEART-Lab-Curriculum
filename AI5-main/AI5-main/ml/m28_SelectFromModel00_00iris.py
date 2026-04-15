import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, accuracy_score
import xgboost as xgb
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, stratify=y, random_state=3377)

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

[79]    validation_0-mlogloss:0.21573
최종 점수 : 0.9666666666666667
acc_score : 0.9666666666666667
[0.08951999 0.06498587 0.4982398  0.34725434]
[0.06498587 0.08951999 0.34725434 0.4982398 ]
Trech=0.065, n=4, ACC: 96.67%
Trech=0.090, n=3, ACC: 96.67%
Trech=0.347, n=2, ACC: 96.67%
Trech=0.498, n=1, ACC: 93.33%
'''