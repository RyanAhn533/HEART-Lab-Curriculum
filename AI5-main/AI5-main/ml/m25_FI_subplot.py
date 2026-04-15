from sklearn.datasets import load_wine, load_digits, load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

random_state = 5656

# 데이터셋 로드
dataset1 = load_breast_cancer()
x1 = dataset1.data
y1 = dataset1.target
dataset2 = load_digits()
x2 = dataset1.data
y2 = dataset1.target
dataset3 = load_wine()
x3 = dataset1.data
y3 = dataset1.target

data_sets = [(x1, y1, 'Breast Cancer'), (x2, y2, 'Digits'), (x3, y3, 'Wine')]

def plot_feature_importances_datasets(model):
            n_features = data_sets.data.shape[1]
            plt.barh(np.arange(n_features), model.feature_importances_,
             align='center')
            plt.yticks(np.arange(n_features),data_sets.feature_names)
            plt.xlabel("Feature Importances")
            plt.ylabel("Features")
            plt.ylim(-1, n_features)
            plt.title(model.__class__.__name__)
    
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
        import matplotlib.pyplot as plt
        import numpy as np
        plot_feature_importances_datasets(model)
        plt.show()
