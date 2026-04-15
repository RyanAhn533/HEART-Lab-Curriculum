from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
# 1. 데이터
datasets = load_diabetes()      # feature_name 때문에
x = datasets.data
y = datasets.target

random_state1=1223
random_state2=1223

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=random_state1)

# 2. 모델 구성
model1 = DecisionTreeRegressor(random_state=random_state1)
model2 = RandomForestRegressor(random_state=random_state1)
model3 = GradientBoostingRegressor(random_state=random_state1)
model4 = XGBRegressor(random_state=random_state1)

models = [model1, model2, model3, model4]

for model in models:
    model.fit(x_train, y_train)
    feature_importances = model.feature_importances_
    
    # 중요도 기반 정렬 (내림차순)
    sorted_idx = np.argsort(feature_importances)
    print('sorted_idx',sorted_idx)
    print(f"\n================= {model.__class__.__name__} =================")
    print('Original R2 Score:', r2_score(y_test, model.predict(x_test)))
    print('Original Feature Importances:', feature_importances)
    
    # 하위 10%, 20%, 30%, 40%, 50% 제거하고 R2 스코어 계산
    for percentage in [10, 20, 30, 40, 50]:
        n_remove = int(len(sorted_idx) * (percentage / 100))
        removed_features_idx = sorted_idx[:n_remove]  # 하위 n% 특성 제거
        print('지운 열의 번호는?', removed_features_idx)
        # 제거된 특성을 제외한 데이터셋 생성
        x_train_reduced = np.delete(x_train, removed_features_idx, axis=1)
        x_test_reduced = np.delete(x_test, removed_features_idx, axis=1)
        
        # 모델 재학습 및 평가
        model.fit(x_train_reduced, y_train)
        r2_reduced = r2_score(y_test, model.predict(x_test_reduced))
        
        print(f"R2 Score after removing {percentage}% lowest importance features: {r2_reduced}")
'''

================= DecisionTreeRegressor =================
Original R2 Score: -0.24733855513252667
Original Feature Importances: [0.05676749 0.01855931 0.23978058 0.08279462 0.05873671 0.0639961
 0.04130515 0.01340568 0.33217096 0.0924834 ]
지운 열의 번호는? [7]
R2 Score after removing 10% lowest importance features: -0.18278951712902836
지운 열의 번호는? [7 1]
R2 Score after removing 20% lowest importance features: -0.14073514732673686
지운 열의 번호는? [7 1 6]
R2 Score after removing 30% lowest importance features: -0.11996383649513986
지운 열의 번호는? [7 1 6 0]
R2 Score after removing 40% lowest importance features: -0.3814711810138871
지운 열의 번호는? [7 1 6 0 4]
R2 Score after removing 50% lowest importance features: -0.163895290380762
sorted_idx [1 7 4 5 0 6 9 3 2 8]

================= RandomForestRegressor =================
Original R2 Score: 0.3687286985683689
Original Feature Importances: [0.05394197 0.00931513 0.25953258 0.1125408  0.04297661 0.05293764
 0.06684433 0.02490964 0.29157054 0.08543076]
지운 열의 번호는? [1]
R2 Score after removing 10% lowest importance features: 0.3519026632903155
지운 열의 번호는? [1 7]
R2 Score after removing 20% lowest importance features: 0.3416184466715577
지운 열의 번호는? [1 7 4]
R2 Score after removing 30% lowest importance features: 0.33406054338438773
지운 열의 번호는? [1 7 4 5]
R2 Score after removing 40% lowest importance features: 0.3189913611109596
지운 열의 번호는? [1 7 4 5 0]
R2 Score after removing 50% lowest importance features: 0.3213144794109841
sorted_idx [1 7 4 0 6 5 9 3 2 8]

================= GradientBoostingRegressor =================
Original R2 Score: 0.3647974813076822
Original Feature Importances: [0.04509096 0.00780692 0.25858035 0.09953666 0.02605597 0.06202725
 0.05303144 0.01840481 0.35346141 0.07600423]
지운 열의 번호는? [1]
R2 Score after removing 10% lowest importance features: 0.33809937503671406
지운 열의 번호는? [1 7]
R2 Score after removing 20% lowest importance features: 0.31546345163096756
지운 열의 번호는? [1 7 4]
R2 Score after removing 30% lowest importance features: 0.32848590831342916
지운 열의 번호는? [1 7 4 0]
R2 Score after removing 40% lowest importance features: 0.32547135160641216
지운 열의 번호는? [1 7 4 0 6]
R2 Score after removing 50% lowest importance features: 0.33372017465187676
sorted_idx [7 0 6 1 3 5 4 9 2 8]

================= XGBRegressor =================
Original R2 Score: 0.10076704957922011
Original Feature Importances: [0.04070464 0.0605858  0.16995801 0.06239288 0.06619858 0.06474677
 0.05363544 0.03795785 0.35376146 0.09005855]
지운 열의 번호는? [7]
R2 Score after removing 10% lowest importance features: 0.185817457646148
지운 열의 번호는? [7 0]
R2 Score after removing 20% lowest importance features: 0.22008345520095707
지운 열의 번호는? [7 0 6]
R2 Score after removing 30% lowest importance features: 0.19327542704911427
지운 열의 번호는? [7 0 6 1]
R2 Score after removing 40% lowest importance features: 0.06483763039416568
지운 열의 번호는? [7 0 6 1 3]
R2 Score after removing 50% lowest importance features: 0.20807068778197846
'''