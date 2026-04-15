import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score
from bayes_opt import BayesianOptimization
import time
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from catboost import CatBoostRegressor

random_state1 = 777

#1. 데이터
path = 'C:\\프로그램\\ai5\\_data\\bike-sharing-demand\\'
# \a  \b 이런걸 하나의 문자로 인식함 줄바꿈 이런거
# # 위와같은 애들 \ -> \\로 바꿔줘야함 / // 도 가능
# path = 'C:/프로그램//ai5\_data\\bike-sharing-demand'

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sampleSubmission_csv = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

train_csv = train_csv.dropna() #train_csv 데이터에서 결측치 삭제
test_csv = test_csv.fillna(test_csv.mean()) #test_csv에는 결측치 평균으로 넣기

x = train_csv.drop(['count'], axis = 1)
y = train_csv[['count']] #, 'registered
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=random_state1, train_size=0.8,
                                                    )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. Bayesian Optimization을 위한 파라미터 설정
bayesian_params = {
    'learning_rate': (0.001, 0.1),
    'depth': (4, 10),
    'l2_leaf_reg': (1, 10),
    'bagging_temperature': (0.0, 1.0),
    'subsample': (0.5, 1),
}

# 3. 블랙박스 함수 정의
def catboost_hamsu(learning_rate, depth, l2_leaf_reg, bagging_temperature, subsample):
    params = {
        'iterations': 100,
        'learning_rate': learning_rate,
        'depth': int(depth),
        'l2_leaf_reg': l2_leaf_reg,
        'bagging_temperature': bagging_temperature,
        'task_type': 'GPU',
        'devices': '0',
        'early_stopping_rounds': 10,
        'verbose': 0
    }
    
    model = CatBoostRegressor(**params)
    model.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=0)
    
    y_predict = model.predict(x_test)
    results = r2_score(y_test, y_predict)
    return results

# 4. Bayesian Optimization 설정
bay = BayesianOptimization(
    f=catboost_hamsu,
    pbounds=bayesian_params,
    random_state=333,
)

# 5. 최적화 실행
n_iter = 100
start_time = time.time()
bay.maximize(init_points=5, n_iter=n_iter)
end_time = time.time()

# 6. 최적 파라미터 및 실행 시간 출력
print(bay.max)
print(f"{n_iter}번 걸린 시간은? {round(end_time - start_time, 2)} 초")

'''
{'target': 0.9992875396796651, 'params': {'bagging_temperature': 0.9965451966939272, 'depth': 6.611701089656843, 'l2_leaf_reg': 1.062853283158693, 'learning_rate': 0.08312989184922419, 'subsample': 
0.6508829834393184}}
100번 걸린 시간은? 83.48 초
'''