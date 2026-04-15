from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
#RandomForest = DecisionTree 앙상블 한거임!
#GradientBoostionClassifier - 경사하강법
from xgboost import XGBClassifier

random_state = 1223
#1. 데이터
dataset = load_iris()
x = dataset.data
y = dataset.target
print(x.shape, y.shape)
#(150, 4) (150,)

from sklearn.model_selection import train_test_split
#나라시 = flatten

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    shuffle=True,
                                                    stratify=y,
                                                    random_state=random_state, train_size=0.8
                                                    )

#2. 모델구성

model1 = DecisionTreeClassifier(random_state=random_state)
model2 = RandomForestClassifier(random_state=random_state)
model3 = GradientBoostingClassifier(random_state=random_state)
model4 = XGBClassifier(random_state=random_state)

models = [model1, model2, model3, model4]
print("random stae 는? ", random_state)
for model in models:
    model.fit(x_train, y_train)
    print("++++++++++++++++++++++++++++++", model.__class__.__name__, "+++++++++++++++++++++++")
    print('acc', model.score(x_test, y_test))
    print(model.feature_importances_)
        
import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances_datasets(model):
    n_features = dataset.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
             align='center')
    plt.yticks(np.arange(n_features),dataset.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)
    plt.title(model.__class__.__name__)

plot_feature_importances_datasets(model)
plt.show()