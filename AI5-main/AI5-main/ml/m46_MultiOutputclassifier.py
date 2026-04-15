import numpy as np
from sklearn.datasets import load_linnerud
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
import warnings
warnings.filterwarnings('ignore')

def create_multiclass_data_with_labels():
    # X 데이터 생성 (20, 3)
    X = np.random.randn(20, 3)
    
    # y 데이터 생성 (20, 3)
    y = np.random.randint(0, 5, size=(20, 3))
    
    # 데이터프레임으로 변환
    X_df = pd.DataFrame(X, columns=['Feature1', 'Feature2', 'Feature3'])
    y_df = pd.DataFrame(y, columns=['Label1', 'Label2', 'Label3'])
    
    return X_df, y_df

# 데이터 생성
x, y = create_multiclass_data_with_labels()
print("X 데이터")
print(x)
print("\nY 데이터")
print(y)

#2. 모델
model = RandomForestClassifier()
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
      round(accuracy_score(y, y_pred), 4))
print(model.predict([[-1.217959, -2.219162, -0.997906]]))



exit()

model = XGBClassifier()
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
      round(accuracy_score(y, y_pred), 4))
print(model.predict([[-1.217959, -2.219162, -0.997906]]))


model = CatBoostClassifier(loss_function='MultiRMSE') #에러난거 MultiRMSE로 해결
model.fit(x,y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
      round(accuracy_score(y, y_pred), 4))
print(model.predict([[-1.217959, -2.219162, -0.997906]]))



model = MultiOutputClassifier(CatBoostRegressor()) #다차원이라서 안된다 MultiOutputRegressor쓰면된다 
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
      round(accuracy_score(y, y_pred), 4))
print(model.predict([[-1.217959, -2.219162, -0.997906]]))


model = MultiOutputClassifier(LGBMClassifier()) #다차원이라서 안된다 MultiOutputRegressor쓰면된다 
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
      round(accuracy_score(y, y_pred), 4))
print(model.predict([[-1.217959, -2.219162, -0.997906]]))
