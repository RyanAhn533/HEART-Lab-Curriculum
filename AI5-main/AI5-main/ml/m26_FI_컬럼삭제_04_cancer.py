from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
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

for model in models:
    model.fit(x_train, y_train)
    feature_importances = model.feature_importances_
    
    # 중요도 기반 정렬 (내림차순)
    sorted_idx = np.argsort(feature_importances)
    print('sorted_idx',sorted_idx)
    print(f"\n================= {model.__class__.__name__} =================")
    print('Original R2 Score:', accuracy_score(y_test, model.predict(x_test)))
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
================= DecisionTreeClassifier =================
Original R2 Score: 0.9473684210526315
Original Feature Importances: [0.         0.05030732 0.         0.         0.         0.
 0.         0.         0.         0.0125215  0.         0.03023319
 0.         0.         0.         0.         0.00785663 0.
 0.         0.         0.72931244 0.         0.0222546  0.01862569
 0.01611893 0.         0.         0.0955152  0.01725451 0.        ]
지운 열의 번호는? [ 0 26 25]
R2 Score after removing 10% lowest importance features: 0.7361111111111112
지운 열의 번호는? [ 0 26 25 21 19 18]
R2 Score after removing 20% lowest importance features: 0.7738095238095238
지운 열의 번호는? [ 0 26 25 21 19 18 17 15 13]
R2 Score after removing 30% lowest importance features: 0.7738095238095238
지운 열의 번호는? [ 0 26 25 21 19 18 17 15 13 12 14 29]
R2 Score after removing 40% lowest importance features: 0.7738095238095238
지운 열의 번호는? [ 0 26 25 21 19 18 17 15 13 12 14 29  8  7  6]
R2 Score after removing 50% lowest importance features: 0.7361111111111112
sorted_idx [ 9  8 11  4 18 12 19 15 17 14 16 29 28  1 24 25 21  0 13 10  5  6 26  2
  3  7 20 23 22 27]

================= RandomForestClassifier =================
Original R2 Score: 0.9298245614035088
Original Feature Importances: [0.02086793 0.01067931 0.04449403 0.05946836 0.00319558 0.02670713
 0.02937097 0.0699787  0.00244273 0.0020165  0.02515119 0.00295319
 0.00331962 0.02345914 0.00409896 0.00361462 0.00490543 0.00386996
 0.00322823 0.00348085 0.12409483 0.0202164  0.14903546 0.1282556
 0.01500504 0.01531022 0.03516301 0.15135923 0.0075333  0.00672446]
지운 열의 번호는? [ 9  8 11]
R2 Score after removing 10% lowest importance features: 0.7361111111111112
지운 열의 번호는? [ 9  8 11  4 18 12]
R2 Score after removing 20% lowest importance features: 0.6984126984126985
지운 열의 번호는? [ 9  8 11  4 18 12 19 15 17]
R2 Score after removing 30% lowest importance features: 0.6984126984126985
지운 열의 번호는? [ 9  8 11  4 18 12 19 15 17 14 16 29]
R2 Score after removing 40% lowest importance features: 0.7361111111111112
지운 열의 번호는? [ 9  8 11  4 18 12 19 15 17 14 16 29 28  1 24]
R2 Score after removing 50% lowest importance features: 0.6984126984126985
sorted_idx [ 4 15  5 16  8  0 25 18 29 17 10 28  2  3 14 13  9 19 12 26 24 11  6 23
  1 21  7 27 22 20]

================= GradientBoostingClassifier =================
Original R2 Score: 0.9385964912280702
Original Feature Importances: [4.15583247e-05 2.49728751e-02 7.60986919e-04 1.11939660e-03
 0.00000000e+00 5.15800222e-06 1.45506477e-02 4.88975719e-02
 1.92724924e-05 1.93825056e-03 5.79516755e-04 8.49675314e-03
 2.65256389e-03 1.53430191e-03 1.23099014e-03 0.00000000e+00
 1.15082117e-05 1.07631999e-04 7.54565070e-05 2.26872696e-03
 5.45900827e-01 3.78644900e-02 1.65739410e-01 2.37203425e-02
 5.99712077e-03 4.95963296e-05 5.16129480e-03 1.05546569e-01
 6.73761776e-04 8.34215822e-05]
지운 열의 번호는? [ 4 15  5]
R2 Score after removing 10% lowest importance features: 0.7361111111111112
지운 열의 번호는? [ 4 15  5 16  8  0]
R2 Score after removing 20% lowest importance features: 0.7361111111111112
지운 열의 번호는? [ 4 15  5 16  8  0 25 18 29]
R2 Score after removing 30% lowest importance features: 0.7361111111111112
지운 열의 번호는? [ 4 15  5 16  8  0 25 18 29 17 10 28]
R2 Score after removing 40% lowest importance features: 0.7361111111111112
지운 열의 번호는? [ 4 15  5 16  8  0 25 18 29 17 10 28  2  3 14]
R2 Score after removing 50% lowest importance features: 0.7361111111111112
sorted_idx [ 2  3  4 25 12 15 10  8 19 16 18  6 17 14  9 28 13 23 21 26 24  0  5 11
 29  1  7 27 22 20]

================= XGBClassifier =================
Original R2 Score: 0.9385964912280702
Original Feature Importances: [0.01410364 0.01792158 0.         0.         0.         0.01414043
 0.00264063 0.05220545 0.00094232 0.00426078 0.00051078 0.01685394
 0.         0.01108845 0.00405639 0.00049153 0.00128015 0.00403847
 0.00186279 0.00101614 0.38543397 0.01304734 0.3068675  0.01161618
 0.01369678 0.         0.01367813 0.08019859 0.0109833  0.01706486]
지운 열의 번호는? [2 3 4]
R2 Score after removing 10% lowest importance features: 0.7361111111111112
지운 열의 번호는? [ 2  3  4 25 12 15]
R2 Score after removing 20% lowest importance features: 0.7361111111111112
지운 열의 번호는? [ 2  3  4 25 12 15 10  8 19]
R2 Score after removing 30% lowest importance features: 0.7361111111111112
지운 열의 번호는? [ 2  3  4 25 12 15 10  8 19 16 18  6]
R2 Score after removing 40% lowest importance features: 0.6984126984126985
지운 열의 번호는? [ 2  3  4 25 12 15 10  8 19 16 18  6 17 14  9]
R2 Score after removing 50% lowest importance features: 0.6984126984126985
'''