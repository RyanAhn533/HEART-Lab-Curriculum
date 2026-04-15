# 23_1 copy

from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# 1. 데이터
datasets = load_breast_cancer()      # feature_name 때문에
x = datasets.data
y = datasets.target

random_state1=1223
random_state2=1223

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, stratify=y, random_state=random_state1)

#2. 모델 구성
model1 = DecisionTreeClassifier(random_state=random_state2)
model2 = RandomForestClassifier(random_state=random_state2)
model3 = GradientBoostingClassifier(random_state=random_state2)
model4 = XGBClassifier(random_state=random_state2)

models = [model1, model2, model3, model4]
random_state = 1223
import matplotlib.pyplot as plt
import numpy as np

models = [model1, model2, model3, model4]
print("random stae 는? ", random_state)
for model in models:
    model.fit(x_train, y_train)
    print("++++++++++++++++++++++++++++++", model.__class__.__name__, "+++++++++++++++++++++++")
    print('acc', model.score(x_test, y_test))
    print(model.feature_importances_)

# print(model)
from xgboost.plotting import plot_importance
plot_importance(model)
plt.show()