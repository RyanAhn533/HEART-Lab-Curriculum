# 23_1 copy

from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.datasets import fetch_california_housing

# 1. 데이터
datasets = load_diabetes()      # feature_name 때문에
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
acc : -0.24733855513252667
[0.05676749 0.01855931 0.23978058 0.08279462 0.05873671 0.0639961
 0.04130515 0.01340568 0.33217096 0.0924834 ]
========== RandomForestRegressor ==========
acc : 0.3687286985683689
[0.05394197 0.00931513 0.25953258 0.1125408  0.04297661 0.05293764
 0.06684433 0.02490964 0.29157054 0.08543076]
========== GradientBoostingRegressor ==========
acc : 0.3647974813076822
[0.04509096 0.00780692 0.25858035 0.09953666 0.02605597 0.06202725
 0.05303144 0.01840481 0.35346141 0.07600423]
========== XGBRegressor ==========
acc : 0.10076704957922011
[0.04070464 0.0605858  0.16995801 0.06239288 0.06619858 0.06474677
 0.05363544 0.03795785 0.35376146 0.09005855]

'''