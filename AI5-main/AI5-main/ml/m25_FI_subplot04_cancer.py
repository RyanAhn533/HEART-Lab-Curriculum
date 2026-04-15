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
random_state : 1223 1223
========== DecisionTreeClassifier ==========
acc : 0.9473684210526315
[0.         0.05030732 0.         0.         0.         0.
 0.         0.         0.         0.0125215  0.         0.03023319 
 0.         0.         0.         0.         0.00785663 0.
 0.         0.         0.72931244 0.         0.0222546  0.01862569 
 0.01611893 0.         0.         0.0955152  0.01725451 0.        ]
========== RandomForestClassifier ==========
acc : 0.9298245614035088
[0.02086793 0.01067931 0.04449403 0.05946836 0.00319558 0.02670713
 0.02937097 0.0699787  0.00244273 0.0020165  0.02515119 0.00295319
 0.00331962 0.02345914 0.00409896 0.00361462 0.00490543 0.00386996
 0.00322823 0.00348085 0.12409483 0.0202164  0.14903546 0.1282556
 0.01500504 0.01531022 0.03516301 0.15135923 0.0075333  0.00672446]
========== GradientBoostingClassifier ==========
acc : 0.9385964912280702
[4.15583247e-05 2.49728751e-02 7.60986919e-04 1.11939660e-03
 0.00000000e+00 5.15800222e-06 1.45506477e-02 4.88975719e-02
 1.92724924e-05 1.93825056e-03 5.79516755e-04 8.49675314e-03
 2.65256389e-03 1.53430191e-03 1.23099014e-03 0.00000000e+00
 1.15082117e-05 1.07631999e-04 7.54565070e-05 2.26872696e-03
 5.45900827e-01 3.78644900e-02 1.65739410e-01 2.37203425e-02
 5.99712077e-03 4.95963296e-05 5.16129480e-03 1.05546569e-01
 6.73761776e-04 8.34215822e-05]
========== XGBClassifier ==========
acc : 0.9385964912280702
[0.01410364 0.01792158 0.         0.         0.         0.01414043
 0.00264063 0.05220545 0.00094232 0.00426078 0.00051078 0.01685394
 0.         0.01108845 0.00405639 0.00049153 0.00128015 0.00403847
 0.00186279 0.00101614 0.38543397 0.01304734 0.3068675  0.01161618
 0.01369678 0.         0.01367813 0.08019859 0.0109833  0.01706486]
'''