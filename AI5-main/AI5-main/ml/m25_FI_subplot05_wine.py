# 23_1 copy

from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# 1. 데이터
datasets = load_wine()      # feature_name 때문에
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

'''
========== DecisionTreeClassifier ==========
acc : 0.8611111111111112
[0.02100275 0.         0.         0.03810533 0.         0.       
 0.13964046 0.         0.         0.         0.03671069 0.3624326
 0.40210817]
========== RandomForestClassifier ==========
acc : 0.9444444444444444
[0.13789135 0.02251876 0.01336314 0.03826336 0.02830375 0.05255915
 0.14261827 0.00916645 0.03234439 0.13563367 0.07199803 0.13963923
 0.17570046]
========== GradientBoostingClassifier ==========
acc : 0.9166666666666666
[1.43202939e-01 4.14470104e-02 5.81184650e-03 1.49084165e-03
 1.36124755e-02 1.91446394e-04 1.13908999e-01 9.45576302e-04
 2.26628207e-04 1.61663632e-01 3.41113315e-03 2.48981160e-01
 2.65106311e-01]
========== XGBClassifier ==========
acc : 0.9444444444444444
[0.07716953 0.03067267 0.04416747 0.00285905 0.01373686 0.0016962
 0.07846211 0.00365221 0.02516203 0.08851561 0.00581782 0.528609
 0.09947944]
'''