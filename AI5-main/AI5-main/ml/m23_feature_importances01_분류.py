#각각의 피처가 성능에 얼마나 기여했는지 명시 할 수 있다.
# 트리 구조의 모델들은 애초에 가지고 있음
# xgboost , dicision tree,  등 등
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
#RandomForest = DecisionTree 앙상블 한거임!
#GradientBoostionClassifier - 경사하강법
from xgboost import XGBClassifier

random_state = 1223
#1. 데이터
x, y = load_iris(return_X_y=True)
print(x, y)
#(150, 4) (150,)
exit()
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
    
''' 
++++++++++++++++++++++++++++++ DecisionTreeClassifier(random_state=777) +++++++++++++++++++++++
acc 0.9666666666666667
[0.03684211 0.         0.06484656 0.89831133]
++++++++++++++++++++++++++++++ RandomForestClassifier(random_state=777) +++++++++++++++++++++++
acc 1.0
[0.09463495 0.02848807 0.40933851 0.46753847]
++++++++++++++++++++++++++++++ GradientBoostingClassifier(random_state=777) +++++++++++++++++++++++
acc 1.0
[0.01160772 0.0100838  0.27084591 0.70746257]
++++++++++++++++++++++++++++++ XGBClassifier(base_score=None, booster=None, callbacks=None,
acc 0.9666666666666667
[0.01255567 0.01918429 0.5991341  0.36912593]
'''

'''
random stae 는?  1223
++++++++++++++++++++++++++++++ DecisionTreeClassifier(random_state=1223) +++++++++++++++++++++++
acc 1.0
[0.01666667 0.         0.57742557 0.40590776]
++++++++++++++++++++++++++++++ RandomForestClassifier(random_state=1223) +++++++++++++++++++++++
acc 1.0
[0.10691492 0.02814393 0.42049394 0.44444721]
++++++++++++++++++++++++++++++ GradientBoostingClassifier(random_state=1223) +++++++++++++++++++++++
acc 1.0
[0.01074646 0.01084882 0.27282247 0.70558224]
acc 1.0
[0.00897023 0.02282782 0.6855639  0.28263798]
'''