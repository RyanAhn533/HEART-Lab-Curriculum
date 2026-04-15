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

x3_datasets = np.array([range(100), range(301, 401,), range(77,177),range(33,133)]).T

y1 = np.array(range(3001, 3101)) #한강의 화씨 온도
y2 = np.array(range(13001, 13101))

print(x1_datasets.shape, y1.shape, y2.shape)

x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1_datasets, x2_datasets,x3_datasets, y1, y2, train_size=0.95, random_state=777,
)
print(x1_train.shape, x2_train.shape, x3_train.shape, x1_test.shape, x2_test.shape, x3_test.shape)
#

#2-1 모델
input1 = Input(shape=(2,))
dense1 = Dense(10, activation='relu', name='bit1')(input1)
dense2 = Dense(20, activation='relu', name='bit2')(dense1)
dense3 = Dense(30, activation='relu', name='bit3')(dense2)
dense4 = Dense(20, activation='relu', name='bit4')(dense3)
output1 = Dense(10, activation='relu', name='bit5')(dense4)
model1 = Model(inputs=input1, outputs=output1)
model1.summary()

#모델2_2
input11 = Input(shape=(3,))
dense11 = Dense(10, activation='relu', name='bit11')(input11)
dense21 = Dense(20, activation='relu', name='bit12')(dense11)
dense31 = Dense(30, activation='relu', name='bit13')(dense21)
dense41 = Dense(20, activation='relu', name='bit14')(dense31)
output11  = Dense(10, activation='relu', name='bit15')(dense41)
model2 = Model(inputs=input11, outputs=output11)
model2.summary()

#모델2_3
input12 = Input(shape=(4,))
dense22 = Dense(10, activation='relu', name='bit22')(input12)
dense32 = Dense(20, activation='relu', name='bit23')(dense22)
dense42 = Dense(30, activation='relu', name='bit23')(dense32)
dense52 = Dense(20, activation='relu', name='bit23')(dense42)
output12  = Dense(10, activation='relu', name='bit24')(dense52)
model2 = Model(inputs=input12, outputs=output12)
model2.summary()

#3. 합체!!!
from keras.layers.merge import concatenate, Concatenate

merge1 = Concatenate(name='mg1')([output1, output11, output12])
merge2 = Dense(5, name='mg2')(merge1)
merge3 = Dense(10, name='mg3')(merge2)
merge4 = Dense(20, name='mg4')(merge3)
merge5 = Dense(10, name='mg5')(merge4)
merge6 = Dense(5, name='mg6')(merge5)

last_output1 = Dense(1, name='last1')(merge6)
last_output2 = Dense(1, name='last2')(merge6)

model = Model(inputs = [input1, input11, input12], outputs=[last_output1, last_output2])

model.summary()

model.compile(loss='mse', optimizer='adam')


import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')

path1 = './_save/keras62/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path1, 'k62_ensemble', date, '_', filename])


mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    verbose=1,
    save_best_only=True,
    filepath=filepath
    )

model.fit([x1_train,x2_train,x3_train],[y1_train, y2_train], epochs=2000, validation_split=0.2, batch_size=1, callbacks=[mcp])

mse = model.evaluate([x1_test, x2_test, x3_test], [y1_test, y2_test], batch_size=1)
print("mse : ", mse)

# 예측
x_predict1 = np.array([range(100, 105), range(401, 406)]).T
x_predict2 = np.array([range(201, 206), range(511, 516), range(249, 254)]).T
x_predict3 = np.array([range(100,105), range(401,406), range(177,182),range(133,138)]).T
y_predict = model.predict([x_predict1, x_predict2, x_predict3])
print("예측 결과: ", y_predict)
loss = model.evaluate([x1_test,x2_test,x3_test], [y1_test, y2_test])
print("로스는 ?", loss)
#로스는 ? [1.3470712900161743, 1.2839874029159546, 0.06308384239673615]
 #로스는 ? [0.0005500555271282792, 0.00018403530702926219, 0.00036602019099518657]