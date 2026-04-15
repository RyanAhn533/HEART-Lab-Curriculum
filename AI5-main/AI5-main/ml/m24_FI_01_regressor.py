#california diabets

from sklearn.datasets import load_diabetes
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
random_state = 5656

# 데이터셋 로드
x1, y1 = load_diabetes(return_X_y=True)
dataset = fetch_california_housing()
x2 = dataset.data
y2 = dataset.target

print(x1.shape, y1.shape, x2.shape, y2.shape)

data_sets = [(x1, y1, 'load_diabetes'), (x2, y2, 'California_housing')]

for x, y, name in data_sets:
    print(f"\n==================== {name} Dataset ====================")
    x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=random_state, train_size=0.8)

    # 모델 구성
    model1 = DecisionTreeRegressor(random_state=random_state)
    model2 = RandomForestRegressor(random_state=random_state)
    model3 = GradientBoostingRegressor(random_state=random_state)
    model4 = XGBRegressor(random_state=random_state)

    models = [model1, model2, model3, model4]

    for model in models:
        model.fit(x_train, y_train)
        print(f"\nModel: {model.__class__.__name__}")
        print('R2 :', model.score(x_test, y_test))
        print('Feature Importances:', model.feature_importances_)

'''
===== load_diabetes Dataset =====

Model: DecisionTreeRegressor
R2 : -0.26987839091059307
Feature Importances: [0.06835323 0.01346413 0.19796712 0.07945736 0.03503489 0.06328675
 0.05043087 0.0107118  0.39203458 0.08925928]

Model: RandomForestRegressor
R2 : 0.3943211142763776
Feature Importances: [0.06434254 0.0126942  0.22485452 0.09873612 0.04631297 0.05636301
 0.05156649 0.02992267 0.34752069 0.06768679]

Model: GradientBoostingRegressor
R2 : 0.40960207905257884
Feature Importances: [0.05703386 0.02205736 0.22656849 0.10289364 0.02640425 0.05227067
 0.0444934  0.02606727 0.38158086 0.0606302 ]

Model: XGBRegressor
R2 : 0.22574749105661107
Feature Importances: [0.03154946 0.06952289 0.13489345 0.06862484 0.03978528 0.0706853
 0.05748887 0.09240093 0.35689822 0.0781508 ]

===== California_housing Dataset =====

Model: DecisionTreeRegressor
R2 : 0.5839514086804332
Feature Importances: [0.52360978 0.05501842 0.05273241 0.03100372 0.0328007  0.13013362
 0.08774068 0.08696066]

Model: RandomForestRegressor
R2 : 0.8108917704643122
Feature Importances: [0.52372292 0.05524638 0.04486304 0.02941652 0.03230679 0.13404268
 0.09016549 0.09023618]

Model: GradientBoostingRegressor
R2 : 0.7926297073205603
Feature Importances: [0.60394133 0.035377   0.01859574 0.00486017 0.00183369 0.12284462
 0.09701968 0.11552778]

Model: XGBRegressor
R2 : 0.8376682135854383
Feature Importances: [0.49307075 0.06969821 0.04587553 0.02455057 0.02236136 0.14518616
 0.08948294 0.10977444]
 
 '''