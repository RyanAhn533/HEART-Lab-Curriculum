from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from keras.layers.merge import concatenate, Concatenate

# 1. 데이터
datasets = fetch_california_housing()
df = pd.DataFrame(data=datasets.data, columns=datasets.feature_names)
x = df.values
y = datasets.target

# 데이터 분할
random_state1 = 1223
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=random_state1)

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
    print(f"\n================= {model.__class__.__name__} =================")
    print('Original R2 Score:', r2_score(y_test, model.predict(x_test)))
    
    # 하위 10%, 20%, 30%, 40%, 50% 제거하고 R2 스코어 계산
    for percentage in [10, 20, 30, 40, 50]:
        n_remove = int(len(sorted_idx) * (percentage / 100))
        removed_features_idx = sorted_idx[:n_remove]  # 하위 n% 특성 제거
                # 제거된 특성을 제외한 데이터셋 생성
        features_idx = sorted_idx[n_remove:-1]  # 하위 n% 특성 제거
        x_train_reduced = np.delete(x_train, removed_features_idx, axis=1)
        x_test_reduced = np.delete(x_test, removed_features_idx, axis=1)
        delted_x_train = np.delete(x_train, features_idx, axis=1)
        delted_x_test = np.delete(x_test, features_idx, axis=1)
        
        n = len(removed_features_idx)
        print(n)
        for i in enumerate (n):
            pca = PCA(n_components=i)
            x_train_pca = pca.fit_transform(delted_x_train)
            x_test_pca = pca.fit_transform(delted_x_test)
            
            x_train = concatenate([x_train_pca, x_train_reduced])
            x_test = concatenate([x_test_pca, x_test_reduced])
            
        # 모델 재학습 및 평가
        model.fit(x_train, y_train)
        r2_pca = r2_score(y_test, model.predict(x_test))
        
        print(f"R2 Score after removing {percentage}% lowest importance features: {r2_pca}")

