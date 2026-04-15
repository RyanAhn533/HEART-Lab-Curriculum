# 23_1 copy

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.datasets import fetch_california_housing

# 1. 데이터
datasets = fetch_california_housing()      # feature_name 때문에
x = datasets.data
y = datasets.target

random_state1=1223
random_state2=1223

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=random_state1)

#2. 모델 구성
model1 = DecisionTreeRegressor(random_state=random_state2)
model2 = RandomForestRegressor(random_state=random_state2)
model3 = GradientBoostingRegressor(random_state=random_state2)
model4 = XGBRegressor(random_state=random_state2)

models = [model1, model2, model3, model4]

import matplotlib.pyplot as plt
import numpy as np

# print(model)

def plot_feature_importances_dataset(model):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1, n_features)
    plt.title(model.__class__.__name__)

print('random_state :', random_state1, random_state2)
for i, model in enumerate(models):
    model.fit(x_train, y_train)
    print('==========', model.__class__.__name__, '==========')
    print('acc :', model.score(x_test, y_test))
    print(model.feature_importances_)
    plt.subplot(2, 2, i+1)
    plot_feature_importances_dataset(model)

plt.rc('xtick', labelsize=5)
plt.rc('ytick', labelsize=5)
plt.tight_layout()      # 간격 안겹치게
plt.show()

'''
========== DecisionTreeRegressor ==========
acc : 0.5964140465722068
[0.51873533 0.05014494 0.05060456 0.02551158 0.02781676 0.13387334
 0.09833673 0.09497676]
========== RandomForestRegressor ==========
acc : 0.811439104037621
[0.52445075 0.05007899 0.04596161 0.03031591 0.03121773 0.1362301
 0.09138102 0.09036389]
========== GradientBoostingRegressor ==========
acc : 0.7865333436969877
[0.60051609 0.02978481 0.02084099 0.00454408 0.0027597  0.12535772
 0.08997582 0.12622079]
========== XGBRegressor ==========
acc : 0.8384930657222394
[0.49375907 0.06520814 0.04559402 0.02538511 0.02146595 0.14413244
 0.0975963  0.10685894]
'''