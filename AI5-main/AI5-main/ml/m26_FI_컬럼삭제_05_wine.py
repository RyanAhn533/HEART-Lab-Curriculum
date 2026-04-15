# 23_1 copy

from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
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
datasets = load_wine()      # feature_name 때문에
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
Original R2 Score: 0.8611111111111112
Original Feature Importances: [0.02100275 0.         0.         0.03810533 0.         0.
 0.13964046 0.         0.         0.         0.03671069 0.3624326
 0.40210817]
지운 열의 번호는? [1]
R2 Score after removing 10% lowest importance features: 0.7715736040609137
지운 열의 번호는? [1 2]
R2 Score after removing 20% lowest importance features: 0.6345177664974619
지운 열의 번호는? [1 2 4]
R2 Score after removing 30% lowest importance features: 0.5888324873096447
지운 열의 번호는? [1 2 4 5 7]
R2 Score after removing 40% lowest importance features: 0.4974619289340101
지운 열의 번호는? [1 2 4 5 7 8]
R2 Score after removing 50% lowest importance features: 0.6802030456852791
sorted_idx [ 7  2  1  4  8  3  5 10  9  0 11  6 12]

================= RandomForestClassifier =================
Original R2 Score: 0.9444444444444444
Original Feature Importances: [0.13789135 0.02251876 0.01336314 0.03826336 0.02830375 0.05255915
 0.14261827 0.00916645 0.03234439 0.13563367 0.07199803 0.13963923
 0.17570046]
지운 열의 번호는? [7]
R2 Score after removing 10% lowest importance features: 0.9086294416243654
지운 열의 번호는? [7 2]
R2 Score after removing 20% lowest importance features: 0.9086294416243654
지운 열의 번호는? [7 2 1]
R2 Score after removing 30% lowest importance features: 0.9086294416243654
지운 열의 번호는? [7 2 1 4 8]
R2 Score after removing 40% lowest importance features: 0.9543147208121827
지운 열의 번호는? [7 2 1 4 8 3]
R2 Score after removing 50% lowest importance features: 0.9086294416243654
sorted_idx [ 5  8  7  3 10  2  4  1  6  0  9 11 12]

================= GradientBoostingClassifier =================
Original R2 Score: 0.9166666666666666
Original Feature Importances: [1.43202939e-01 4.14470104e-02 5.81184650e-03 1.49084165e-03
 1.36124755e-02 1.91446394e-04 1.13908999e-01 9.45576302e-04
 2.26628207e-04 1.61663632e-01 3.41113315e-03 2.48981160e-01
 2.65106311e-01]
지운 열의 번호는? [5]
R2 Score after removing 10% lowest importance features: 0.8629441624365481
지운 열의 번호는? [5 8]
R2 Score after removing 20% lowest importance features: 0.8629441624365481
지운 열의 번호는? [5 8 7]
R2 Score after removing 30% lowest importance features: 0.8629441624365481
지운 열의 번호는? [ 5  8  7  3 10]
R2 Score after removing 40% lowest importance features: 0.8629441624365481
지운 열의 번호는? [ 5  8  7  3 10  2]
R2 Score after removing 50% lowest importance features: 0.8629441624365481
sorted_idx [ 5  3  7 10  4  8  1  2  0  6  9 12 11]

================= XGBClassifier =================
Original R2 Score: 0.9444444444444444
Original Feature Importances: [0.07716953 0.03067267 0.04416747 0.00285905 0.01373686 0.0016962
 0.07846211 0.00365221 0.02516203 0.08851561 0.00581782 0.528609
 0.09947944]
지운 열의 번호는? [5]
R2 Score after removing 10% lowest importance features: 0.9086294416243654
지운 열의 번호는? [5 3]
R2 Score after removing 20% lowest importance features: 0.9086294416243654
지운 열의 번호는? [5 3 7]
R2 Score after removing 30% lowest importance features: 0.9086294416243654
지운 열의 번호는? [ 5  3  7 10  4]
R2 Score after removing 40% lowest importance features: 0.9086294416243654
지운 열의 번호는? [ 5  3  7 10  4  8]
R2 Score after removing 50% lowest importance features: 0.9086294416243654
    '''