from sklearn.datasets import load_wine, load_digits, load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
random_state = 5656

# 데이터셋 로드
x1, y1 = load_breast_cancer(return_X_y=True)
x2, y2 = load_digits(return_X_y=True)
x3, y3 = load_wine(return_X_y=True)
dataset = [(x1, y1, 'Breast Cancer'), (x2, y2, 'Digits'), (x3, y3, 'Wine')]

for x, y, name in data_sets:
    print(f"\n===== {name} Dataset =====")
    x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=random_state, train_size=0.8)

    # 모델 구성
    model1 = DecisionTreeClassifier(random_state=random_state)
    model2 = RandomForestClassifier(random_state=random_state)
    model3 = GradientBoostingClassifier(random_state=random_state)
    model4 = XGBClassifier(random_state=random_state)

    models = [model1, model2, model3, model4]

    for model in models:
        model.fit(x_train, y_train)
        print(f"\nModel: {model.__class__.__name__}")
        print('Accuracy:', model.score(x_test, y_test))
        print('Feature Importances:', model.feature_importances_)

def plot_feature_importances_datasets(model):
    n_features = dataset.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
             align='center')
    plt.yticks(np.arange(n_features),dataset.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)
    plt.title(model.__class__.__name__)
    
plt.subplot(2,2,1)
plot_feature_importances_datasets(model1)