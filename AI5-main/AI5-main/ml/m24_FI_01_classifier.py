# cancer
# wine
#digits

from sklearn.datasets import load_wine, load_digits, load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
random_state = 5656
x1,y1 = load_breast_cancer(return_X_y=True)
x2,y2 = load_digits(return_X_y=True)
x3,y3 = load_wine(return_X_y=True)
data_sets = [[x1,y1],[x2,y2],[x3,y3]]
#print(data_sets)
range = [1,2,3]
for i in data_sets:
    x_train, x_test, y_train, y_test = train_test_split(i,shuffle=True, random_state=random_state, train_size=0.8 )

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
        
