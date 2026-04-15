from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import sklearn as sk
from sklearn.datasets import fetch_california_housing
import numpy as np
import time as t
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
from tensorflow.keras.callbacks import EarlyStopping

dataset = fetch_california_housing()
print(dataset)
print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target


print(x.shape)
print(y.shape)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,
                                                    shuffle=True, random_state=3)

from sklearn.preprocessing import MinMaxScaler, StandardScaler,MaxAbsScaler, RobustScaler
#scaler = MaxAbsScaler()
#scaler = StandardScaler()
scaler = RobustScaler()
print(x_train)
print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))

#scaler = MaxAbsScaler() 0.61 r2 0.49
#scaler = RobustScaler() r2 0.304  loss 0.91951
#모델

"""
model = Sequential()
model.add(Dense(128, input_dim=8, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

#컴파일 훈련
model.compile(
    loss='mse',
    optimizer='adam',
    metrics=['acc'])
es= EarlyStopping(monitor='val_loss', mode = 'min', patience=20,
                  restore_best_weights=True)
import datetime
date = datetime.datetime.now()
print(date)     # 2024-07-26 16:49:48.004109
print(type(date))   # <class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")
print(date)     # 0726_1655
print(type(date))   # <class 'str'>

path = './_save/keras29_mcp/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 1000-0.7777.hdf56 형태로 파일 명 생성, fit에서의 반환값을 가져옴
filepath = "".join([path, 'k29_', date, '_', filename]) 

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
mcp = ModelCheckpoint(
    monitor='val_loss', 
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath=filepath)
    
model.fit(x_train, y_train, epochs=1000, verbose = 2, 
                 batch_size=32, validation_split=0.3,
                 callbacks=[es,mcp]) # valtidation_data=(x_val, y_val)

"""
end  = t.time()

model = load_model("./_save/keras29_mcp/k29_0729_0932_0020-1.3370.hdf5")

#평가예측
loss = model.evaluate(x_test, y_test)
print("로스는 ?", loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2스코어는? ", r2)


#스케일링 전 0.2713
#스케일링 후 0.7406029396628281
"""

exit()
y_submit = model.predict(test_csv)
samplesubmission_csv['target'] = np.round(y_submit)/10
samplesubmission_csv.to_csv(path + "santafe_7.csv")


로스는 ? [0.2712891101837158, 0.0021802326664328575]
r2스코어는?  0.7948835567807272

로스는 ? [0.3928057849407196, 0.0021802326664328575]
r2스코어는?  0.7030072751830784
Minmax win
"""