# 23_1 copy

from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
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
# 1. 데이터
datasets = load_digits()      # feature_name 때문에
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
sorted_idx [ 0 57 56 55 48 47 40 39 32 62 25 23 16 15 31  7  8  6 14 22 24 58 35 49
 61 11 41 50 52 17 34 51 46  2  4 63 59 19 45  1 18 30 10 38 12  3  9 29
 53 13 44 54 37 20  5 28 27 43 60 33 42 26 36 21]

================= DecisionTreeClassifier =================
Original R2 Score: 0.8472222222222222
Original Feature Importances: [0.         0.00886642 0.00544332 0.01521574 0.00567348 0.04850847
 0.         0.         0.         0.0155011  0.01128834 0.00283519
 0.01376369 0.01847882 0.00077323 0.         0.         0.00441178
 0.00988328 0.00842401 0.04104339 0.08596583 0.00077323 0.
 0.00103098 0.         0.07540495 0.05639249 0.04959762 0.01583598
 0.01060644 0.         0.         0.06354131 0.00462313 0.00139182
 0.07712915 0.03225891 0.01308039 0.         0.         0.00292356
 0.0745241  0.05873763 0.02066762 0.00885585 0.00535316 0.
 0.         0.00152153 0.00341824 0.00489287 0.00391772 0.01633694
 0.02674242 0.         0.         0.         0.00115985 0.00601834
 0.05943809 0.00201041 0.         0.00573917]
지운 열의 번호는? [ 0 57 56 55 48 47]
R2 Score after removing 10% lowest importance features: 0.6100188445755081
지운 열의 번호는? [ 0 57 56 55 48 47 40 39 32 62 25 23]
R2 Score after removing 20% lowest importance features: 0.6550427106097594
지운 열의 번호는? [ 0 57 56 55 48 47 40 39 32 62 25 23 16 15 31  7  8  6 14]
R2 Score after removing 30% lowest importance features: 0.6408246476515747
지운 열의 번호는? [ 0 57 56 55 48 47 40 39 32 62 25 23 16 15 31  7  8  6 14 22 24 58 35 49
 61]
R2 Score after removing 40% lowest importance features: 0.6848329377602413
지운 열의 번호는? [ 0 57 56 55 48 47 40 39 32 62 25 23 16 15 31  7  8  6 14 22 24 58 35 49
 61 11 41 50 52 17 34 51]
R2 Score after removing 50% lowest importance features: 0.6188205025972414
sorted_idx [ 0 56 39 32 24 40 16 31  8 48 47 23 15  7 55  1 57 49 63 14 17 11  4  3
 22  6 59  9 41 25 52 50 45 12 62 35 37 10  5 51 44 46  2 29 18 19 38 27
 54 58 60 13 53 30 33 34 61 20 28 42 36 26 43 21]

================= RandomForestClassifier =================
Original R2 Score: 0.9777777777777777
Original Feature Importances: [0.00000000e+00 1.46684371e-03 2.15828252e-02 8.63100042e-03
 7.74184369e-03 2.11484828e-02 9.75321481e-03 6.78704204e-04
 9.99062839e-05 1.05410820e-02 2.10193342e-02 7.69191223e-03
 1.81855748e-02 2.60401050e-02 5.48039124e-03 5.48899644e-04
 3.24578083e-05 6.79716398e-03 2.34860951e-02 2.37832862e-02
 3.03543272e-02 5.15856136e-02 8.81822788e-03 2.96316379e-04
 2.55452052e-05 1.23990699e-02 4.38164383e-02 2.48623827e-02
 3.31805152e-02 2.20501425e-02 2.71615704e-02 4.56565143e-05
 0.00000000e+00 2.85024358e-02 2.85417748e-02 1.85314325e-02
 3.94845951e-02 2.02555900e-02 2.41710385e-02 0.00000000e+00
 2.82560661e-05 1.08591464e-02 3.44453029e-02 4.44882149e-02
 2.14025504e-02 1.74062193e-02 2.14496422e-02 1.37295796e-04
 1.23540702e-04 2.39008914e-03 1.72195928e-02 2.13446288e-02
 1.33219640e-02 2.63532308e-02 2.50370391e-02 1.21113019e-03
 0.00000000e+00 2.03549096e-03 2.50731385e-02 1.02211980e-02
 2.51334337e-02 2.97603141e-02 1.84862656e-02 3.28051993e-03]
지운 열의 번호는? [ 0 56 39 32 24 40]
R2 Score after removing 10% lowest importance features: 0.9444818494013745
지운 열의 번호는? [ 0 56 39 32 24 40 16 31  8 48 47 23]
R2 Score after removing 20% lowest importance features: 0.9140145716338359
지운 열의 번호는? [ 0 56 39 32 24 40 16 31  8 48 47 23 15  7 55  1 57 49 63]
R2 Score after removing 30% lowest importance features: 0.9316178876773027
지운 열의 번호는? [ 0 56 39 32 24 40 16 31  8 48 47 23 15  7 55  1 57 49 63 14 17 11  4  3
 22]
R2 Score after removing 40% lowest importance features: 0.9329719889114154
지운 열의 번호는? [ 0 56 39 32 24 40 16 31  8 48 47 23 15  7 55  1 57 49 63 14 17 11  4  3
 22  6 59  9 41 25 52 50]
R2 Score after removing 50% lowest importance features: 0.9140145716338359
sorted_idx [ 0 40 39 32 23 48 16 57 31 24 47 55  7 59 56 11  8 25  1 15 14 49  9  4
  3 17 34 35 50  6 61 22 13 41 52 63  2 44 12 37 30 53 58 51 38 18 27 20
 45 10 62 29 54 46 28 19 26 60  5 33 43 36 42 21]

================= GradientBoostingClassifier =================
Original R2 Score: 0.9638888888888889
Original Feature Importances: [0.00000000e+00 9.43576218e-04 8.27093418e-03 2.99739028e-03
 2.97008861e-03 5.88988652e-02 5.62951522e-03 4.34895132e-04
 8.88357015e-04 1.95155677e-03 2.09891586e-02 6.91858455e-04
 9.24903709e-03 7.02342073e-03 1.07147802e-03 9.52195665e-04
 8.50885455e-05 3.76139301e-03 1.66091111e-02 3.89863704e-02
 1.91841797e-02 8.88369854e-02 6.61116977e-03 2.37200119e-07
 1.34856142e-04 9.39554807e-04 4.67291318e-02 1.70291837e-02
 3.59639663e-02 2.54396537e-02 1.23329945e-02 1.21625408e-04
 0.00000000e+00 5.92042227e-02 4.25109611e-03 4.85830972e-03
 7.20654432e-02 9.73968712e-03 1.60211041e-02 0.00000000e+00
 0.00000000e+00 7.10152985e-03 7.93040245e-02 7.13910682e-02
 9.18213134e-03 2.09073603e-02 2.88972048e-02 2.95059335e-04
 2.73206342e-06 1.10655416e-03 5.54005762e-03 1.59939737e-02
 7.94602878e-03 1.39723083e-02 2.80811998e-02 3.48132723e-04
 6.15332082e-04 8.99662610e-05 1.51837762e-02 5.91428210e-04
 5.62177231e-02 5.95652996e-03 2.14412261e-02 7.96696088e-03]
지운 열의 번호는? [ 0 40 39 32 23 48]
R2 Score after removing 10% lowest importance features: 0.9113063691656104
지운 열의 번호는? [ 0 40 39 32 23 48 16 57 31 24 47 55]
R2 Score after removing 20% lowest importance features: 0.9268785333579077
지운 열의 번호는? [ 0 40 39 32 23 48 16 57 31 24 47 55  7 59 56 11  8 25  1]
R2 Score after removing 30% lowest importance features: 0.9211236031129283
지운 열의 번호는? [ 0 40 39 32 23 48 16 57 31 24 47 55  7 59 56 11  8 25  1 15 14 49  9  4
  3]
R2 Score after removing 40% lowest importance features: 0.9089366920059129
지운 열의 번호는? [ 0 40 39 32 23 48 16 57 31 24 47 55  7 59 56 11  8 25  1 15 14 49  9  4
  3 17 34 35 50  6 61 22]
R2 Score after removing 50% lowest importance features: 0.9201080271873436
sorted_idx [ 0 57 56 55 48 47 40 39 32 24 23 16 31  7  8 14 50 11 22 59  4 25 49  6
  2 34  9 17  3 35 18 44 20 27 41 30 13 12 10 52 61 37 51 15 53 45 29 58
 46 54  1 63 62 28 26 42 43 38 19  5 21 36 60 33]

================= XGBClassifier =================
Original R2 Score: 0.9694444444444444
Original Feature Importances: [0.         0.03260561 0.00599625 0.00737296 0.00518664 0.04148781
 0.00560644 0.         0.         0.00720591 0.0132025  0.00289933
 0.01232869 0.01128461 0.00088012 0.01497376 0.         0.00730367
 0.00837447 0.04088178 0.01003765 0.04985778 0.00374498 0.
 0.         0.00520189 0.03477967 0.01006788 0.03358783 0.02671578
 0.01016452 0.         0.         0.07282628 0.00650188 0.00796701
 0.05700776 0.01347518 0.03782811 0.         0.         0.0100892
 0.03535675 0.03744606 0.00969996 0.01911054 0.02964951 0.
 0.         0.00542436 0.00263066 0.01368503 0.01321433 0.01782674
 0.03249998 0.         0.         0.         0.02799876 0.00390022
 0.06468432 0.01331835 0.03346457 0.03264587]
지운 열의 번호는? [ 0 57 56 55 48 47]
R2 Score after removing 10% lowest importance features: 0.8957342049733129
지운 열의 번호는? [ 0 57 56 55 48 47 40 39 32 24 23 16]
R2 Score after removing 20% lowest importance features: 0.8957342049733129
지운 열의 번호는? [ 0 57 56 55 48 47 40 39 32 24 23 16 31  7  8 14 50 11 22]
R2 Score after removing 30% lowest importance features: 0.8845628697918821
지운 열의 번호는? [ 0 57 56 55 48 47 40 39 32 24 23 16 31  7  8 14 50 11 22 59  4 25 49  6
  2]
R2 Score after removing 40% lowest importance features: 0.8953956796647847
지운 열의 번호는? [ 0 57 56 55 48 47 40 39 32 24 23 16 31  7  8 14 50 11 22 59  4 25 49  6
  2 34  9 17  3 35 18 44]
R2 Score after removing 50% lowest importance features: 0.8815161420151283
'''