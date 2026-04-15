from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd

# 1. 데이터
datasets = fetch_california_housing()# feature_name 때문에
print(datasets.feature_names)
['MedInc', 'HouseAge', 'AveRooms', 
 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

df = pd.DataFrame(data=datasets.data, columns=datasets.feature_names)
x = df.values
y = datasets.target

#데이터 분할
random_state1 = 1223
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=random_state1)

# 2. 모델 구성
model1 = DecisionTreeRegressor(random_state=random_state1)
model2 = RandomForestRegressor(random_state=random_state1)
model3 = GradientBoostingRegressor(random_state=random_state1)
model4 = XGBRegressor(random_state=random_state1)

models = [model1, model2, model3, model4]

for i in models:
    i.fit(x_train, y_train)
    feature_importancse = i.feature_importances_
    # 중요도 기반 정렬 (내림차순)
    sorted_idx = np.argsort(feature_importancse)
    