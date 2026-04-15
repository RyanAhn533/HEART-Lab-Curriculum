# 23_1 copy

from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# 1. 데이터
datasets = load_digits()      # feature_name 때문에
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