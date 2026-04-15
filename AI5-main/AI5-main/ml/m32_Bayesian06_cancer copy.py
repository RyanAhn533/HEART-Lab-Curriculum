import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from bayes_opt import BayesianOptimization
import time
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터 로드 및 전처리
x, y = load_breast_cancer(return_X_y=True)
random_state = 777
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=random_state, train_size=0.8, stratify=y)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. BayesianOptimization을 위한 파라미터 설정
bayesian_params = {
    'learning_rate': (0.001, 0.1),
    'max_depth': (3, 10),
}

# 3. 블랙박스 함수 정의
def xgb_hamsu(learning_rate, max_depth):
    params = {
        'n_estimators': 100,
        'learning_rate': learning_rate,
        'max_depth': int(round(max_depth)),  # max_depth는 정수형으로 설정
    }
    
    model = XGBClassifier(**params, n_jobs=-1)
    model.fit(x_train, y_train,
              eval_set=[(x_test, y_test)],
              eval_metric='logloss',
              verbose=0)
    
    y_predict = model.predict(x_test)
    results = accuracy_score(y_test, y_predict)
    return results

# 4. Bayesian Optimization 설정
bay = BayesianOptimization(
    f=xgb_hamsu,
    pbounds=bayesian_params,
    random_state=333,
)

# 5. 최적화 실행
n_iter = 100
start_time = time.time()
bay.maximize(init_points=5, n_iter=n_iter)  # 'inin_points'를 'init_points'로 수정
end_time = time.time()

# 6. 최적 파라미터 및 실행 시간 출력
print(bay.max)
print(f"{n_iter}번 걸린 시간은? {round(end_time - start_time, 2)} 초")
