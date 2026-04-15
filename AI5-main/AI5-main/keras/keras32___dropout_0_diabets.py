from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import sklearn as sk
from sklearn.datasets import load_diabetes
import numpy as np
import time as t
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
from tensorflow.keras.callbacks import EarlyStopping

dataset = load_diabetes()
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
scaler = RobustScaler()
#MaxAbsScaler 2980 0.44
#RobustScaler r2스코어는?  0.41430952830512535 로스는 ? [3171.337158203125
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


print(x_train)
print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))


#모델
model = Sequential()
model.add(Dense(128, input_dim=10, activation='relu'))
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

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
mcp = ModelCheckpoint(
    monitor='val_loss', 
    mode='auto',
    verbose=1,
    save_best_only=True, filepath=("./_save/keras30/keras30_3_save_model.h1"))

model.fit(x_train, y_train, epochs=200, batch_size=128,
          verbose=1, validation_split=0.2, callbacks=[es])
#model.save("./_save/keras30/keras30_3")
model = load_model("./_save/keras30/keras30_3")
#평가예측
loss = model.evaluate(x_test, y_test)
print("로스는 ?", loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2스코어는? ", r2)


#스케일링 전 0.3892
#스케일링 후 0.40
"""

exit()
y_submit = model.predict(test_csv)
samplesubmission_csv['target'] = np.round(y_submit)/10
samplesubmission_csv.to_csv(path + "santafe_7.csv")

로스는 ? [3356.566650390625, 0.0]
r2스코어는?  0.38010082669156087

로스는 ? [3220.5322265625, 0.0]
r2스코어는?  0.4052240583681187

Minmax WIn
"""