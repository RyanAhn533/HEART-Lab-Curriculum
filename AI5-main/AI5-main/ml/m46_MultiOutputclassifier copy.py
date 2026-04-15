import numpy as np
from sklearn.datasets import load_linnerud
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
import warnings
warnings.filterwarnings('ignore')

# 난수 고정
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# 데이터 생성 함수
def create_multiclass_data_with_labels():
    # X 데이터 생성 (20, 3)
    X = np.random.randn(20, 3)
    
    # y 데이터 생성 (20, 3)
    y = np.random.randint(0, 5, size=(20, 3))
    
    # 데이터프레임으로 변환
    X_df = pd.DataFrame(X, columns=['Feature1', 'Feature2', 'Feature3'])
    y_df = pd.DataFrame(y, columns=['Label1', 'Label2', 'Label3'])
    
    return X_df.values, y_df.values

# 데이터 생성
x, y = create_multiclass_data_with_labels()
print("X 데이터")
print(x)
print("\nY 데이터")
print(y)
print(x.shape, y.shape)
#(20, 3) (20, 3)

#2. 모델
model = RandomForestClassifier()
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4))
print(model.predict([[ 0.49671415, -0.1382643,   0.64768854]]))

model = MultiOutputClassifier(XGBClassifier())
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4))
print(model.predict([[ 0.49671415, -0.1382643,   0.64768854]]))


model = MultiOutputClassifier(CatBoostClassifier())
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred.reshape(20,3)), 4))
print(model.predict([[ 0.49671415, -0.1382643,   0.64768854]]))

model = MultiOutputClassifier(LGBMClassifier()) #다차원이라서 안된다 MultiOutputRegressor쓰면된다 
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4))
print(model.predict([[ 0.49671415, -0.1382643,   0.64768854]]))
