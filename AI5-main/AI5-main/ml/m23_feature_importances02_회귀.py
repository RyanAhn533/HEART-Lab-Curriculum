##회귀##
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

#RandomForest = DecisionTree 앙상블 한거임!
#GradientBoostionClassifier - 경사하강법
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split


random_state = 1223
#1. 데이터
dataset = load_diabetes()
x = dataset.data
y = dataset.target
print(x.shape, y.shape)
#(442, 10) (442,)

#나라시 = flatten

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    shuffle=True,
                                                    random_state=random_state, train_size=0.8,
                                                    #stratify=y
                                                    )

#2. 모델구성

model1 = DecisionTreeRegressor(random_state=random_state)
model2 = RandomForestRegressor(random_state=random_state)
model3 = GradientBoostingRegressor(random_state=random_state)
model4 = XGBRegressor(random_state=random_state)

models = [model1, model2, model3, model4]
print("random stae 는? ", random_state)
for model in models:
    model.fit(x_train, y_train)
    print("++++++++++++++++++++++++++++++", model.__class__.__name__, "+++++++++++++++++++++++")
    print('acc', model.score(x_test, y_test))
    print(model.feature_importances_)
    
'''
random stae 는?  1223
++++++++++++++++++++++++++++++ DecisionTreeRegressor +++++++++++++++++++++++
acc -0.24733855513252667
[0.05676749 0.01855931 0.23978058 0.08279462 0.05873671 0.0639961
 0.04130515 0.01340568 0.33217096 0.0924834 ]
++++++++++++++++++++++++++++++ RandomForestRegressor +++++++++++++++++++++++
acc 0.3687286985683689
[0.05394197 0.00931513 0.25953258 0.1125408  0.04297661 0.05293764
 0.06684433 0.02490964 0.29157054 0.08543076]
++++++++++++++++++++++++++++++ GradientBoostingRegressor +++++++++++++++++++++++
acc 0.3647974813076822
[0.04509096 0.00780692 0.25858035 0.09953666 0.02605597 0.06202725
 0.05303144 0.01840481 0.35346141 0.07600423]
++++++++++++++++++++++++++++++ XGBRegressor +++++++++++++++++++++++
acc 0.10076704957922011
[0.04070464 0.0605858  0.16995801 0.06239288 0.06619858 0.06474677
 0.05363544 0.03795785 0.35376146 0.09005855]
 '''
 
 #4000개의 컬럼에서 나머지 2000개는 별로야
 # 2~3천개를 PCA로 병합
 #아니면 삭제 !
 # -1 ~ 1 까지에 다양한 상관관계가 있음
 # 상관관계가 1이면 weight 정방향으로 연관이 있다, 상관관계가 -1이면 weight 반대방향으로 연관
 # 상관관계 값이 0인게 제일 안좋음
 #1과 -1이면 과적합 된다.
 
  