import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import sequential, Model
from keras.layers import Dense, InputLayer, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
x1_datasets = np.array([range(100), range(301, 401)]).T #1by100
# print(x1_datasets)
# print(x1_datasets.shape)
# # x1_datasets = 삼성 종가, 하이닉스 종가
# print('#######################')
x2_datasets = np.array([range(101, 201), range(411, 511), range(150, 250)]).transpose() #3by100
# print(x2_datasets)
# print(x2_datasets.shape)
#원유, 환율, 금시세

y = np.array(range(3001, 3101)) #한강의 화씨 온도



#x_train1, x_test1, y_train1, y_test1 = train_test_split(x1_datasets, y, train_size=0.8, shuffle=True, random_state=5656)
#x_train2, x_test2, y_train2, y_test2 = train_test_split(x2_datasets, y, train_size=0.8, shuffle=True, random_state=5656)

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
    x1_datasets, x2_datasets, y, train_size=0.95, random_state=777,
)

# print(x1_train.shape, x2_train.shape, y_train.shape)

#2-1 모델
input1 = Input(shape=(2,))
dense1 = Dense(10, activation='relu', name='bit1')(input1)
dense2 = Dense(20, activation='relu', name='bit2')(dense1)
dense3 = Dense(30, activation='relu', name='bit3')(dense2)
dense4 = Dense(20, ac
               tivation='relu', name='bit4')(dense3)
output1 = Dense(10, activation='relu', name='bit5')(dense4)
model1 = Model(inputs=input1, outputs=output1)
model1.summary()

#모델2
input11 = Input(shape=(3,))
dense11 = Dense(100, activation='relu', name='bit11')(input11)
dense21 = Dense(200, activation='relu', name='bit12')(dense11)
output11  = Dense(100, activation='relu', name='bit13')(dense21)
model2 = Model(inputs=input11, outputs=output11)
model2.summary()

#2-3. 합체!!!
from keras.layers.merge import concatenate, Concatenate
#merge가 뭐지? -> 합치다
# merge1 = concatenate([dense4, dense21])
# model1 = Dense(10)(merge1)
# model2 = Dense(5)(model1)
# output = Dense(1)(model2)
# model = Model(inputs = [input1, input11], outputs = output)
# model.summary()

merge1 = Concatenate(name='mg1')([output1, output11])
merge2 = Dense(5, name='mg2')(merge1)
merge3 = Dense(10, name='mg3')(merge2)
merge4 = Dense(5, name='mg3')(merge3)
last_output = Dense(1, name='last')(merge3)

model = Model(inputs = [input1, input11], outputs=last_output)
model.summary()

model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(
    monitor='val_loss',
    mode = 'min',
    patience=30,
    verbose=1,
    restore_best_weights=True
)

import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')

path1 = './_save/keras62/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path1, 'k62_ensemble1_', date, '_', filename])


mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    verbose=1,
    save_best_only=True,
    filepath=filepath
    )

model.fit([x1_train,x2_train],y_train, epochs=2000, validation_split=0.2, batch_size=8, callbacks=[es, mcp])
"""
mse = model.evaluate([x1_test, x2_test], y_test, batch_size=1)
print("mse : ", mse)

x_predict1 = np.array([range(100,105), range(401, 405)]).T
x_predict2 = np.array([range(201, 205), range(511, 515), range(250, 255)]).transpose()

y_predict = model.predict([x_predict1, x_predict2])

"""

mse = model.evaluate([x1_test, x2_test], y_test, batch_size=2)
print("mse : ", mse)

# 예측
x_predict1 = np.array([range(100, 105), range(401, 406)]).T
x_predict2 = np.array([range(201, 206), range(511, 516), range(249, 254)]).T

y_predict = model.predict([x_predict1, x_predict2])
print("예측 결과: ", y_predict)
loss = model.evaluate([x1_test,x2_test], y_test)
print("로스는 ?", loss)
# 예측 결과:  [[3104.794 ]
#  [3109.4023]
#  [3114.1243]
#  [3119.0913]
#  [3124.0923]]

# mse :  0.00013172626495361328
# 예측 결과:  [[3105.731 ]
#  [3110.5098]
#  [3115.2888]
#  [3120.0671]
#  [3124.8958]]
# 1/1 [==============================] - 0s 11ms/step - loss: 1.3173e-04
# 로스는 ? 0.00013172626495361328