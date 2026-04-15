import numpy as np
from sklearn.datasets import load_linnerud
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
import warnings
warnings.filterwarnings('ignore')
#1. 데이터
x, y = load_linnerud(return_X_y=True)
#print(x.shape, y.shape) #(20, 3) (20, 3)
#print(x)
'''
[[  5. 162.  60.]
 [  2. 110.  60.]
 [ 12. 101. 101.]
 [ 12. 105.  37.]
 [ 13. 155.  58.]
 [  4. 101.  42.]
 [  8. 101.  38.]
 [  6. 125.  40.]
 [ 15. 200.  40.]
 [ 17. 251. 250.]
 [ 17. 120.  38.]
 [ 13. 210. 115.]
 [ 14. 215. 105.]
 [  1.  50.  50.]
 [  6.  70.  31.]
 [ 12. 210. 120.]
 [  4.  60.  25.]
 [ 11. 230.  80.]
 [ 15. 225.  73.]
 [  2. 110.  43.]]
'''
#print(y)
'''
[[191.  36.  50.]
 [189.  37.  52.]
 [193.  38.  58.]
 [162.  35.  62.]
 [189.  35.  46.]
 [182.  36.  56.]
 [211.  38.  56.]
 [167.  34.  60.]
 [176.  31.  74.]
 [154.  33.  56.]
 [169.  34.  50.]
 [166.  33.  52.]
 [154.  34.  64.]
 [247.  46.  50.]
 [193.  36.  46.]
 [202.  37.  62.]
 [176.  37.  54.]
 [157.  32.  52.]
 [156.  33.  54.]
 [138.  33.  68.]]
'''

#2. 모델
model = RandomForestRegressor()
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4))
print(model.predict([[2, 110, 43]]))

# RandomForestRegressor 스코어 :  3.6693
# [[153.55  34.07  63.84]]

model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4))
print(model.predict([[2, 110, 43]]))
# LinearRegression 스코어 :  7.4567
# [[187.33745435  37.08997099  55.40216714]]

model = Ridge()
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4))
print(model.predict([[2, 110, 43]]))
# Ridge 스코어 :  7.4569
# [[187.32842123  37.0873515   55.40215097]]

model = XGBRegressor()
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4))
print(model.predict([[2, 110, 43]]))
# XGBRegressor 스코어 :  0.0008
# [[138.0005    33.002136  67.99897 ]]

model = CatBoostRegressor(loss_function='MultiRMSE') #에러난거 MultiRMSE로 해결
model.fit(x,y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4))
print(model.predict([[2, 110, 43]]))
# CatBoostRegressor 스코어 :  0.0638
# [[138.21649371  32.99740595  67.8741709 ]]


model = MultiOutputRegressor(CatBoostRegressor()) #다차원이라서 안된다 MultiOutputRegressor쓰면된다 
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4))
print(model.predict([[2, 110, 43]]))
# MultiOutputRegressor 스코어 :  0.2154
# [[138.97756017  33.09066774  67.61547996]]


model = MultiOutputRegressor(LGBMRegressor()) #다차원이라서 안된다 MultiOutputRegressor쓰면된다 
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4))
print(model.predict([[2, 110, 43]]))
# MultiOutputRegressor 스코어 :  8.91
# [[178.6  35.4  56.1]]

