import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import sequential, Model
from keras.layers import Dense, InputLayer, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
x_datasets = np.array([range(100), range(301, 401)]).T #1by100

y1 = np.array(range(3001, 3101)) #한강의 화씨 온도
y2 = np.array(range(13001, 13101))

#2-1 모델
input1 = Input(shape=(2,))
dense1 = Dense(10, activation='relu', name='bit1')(input1)
dense2 = Dense(20, activation='relu', name='bit2')(dense1)
dense3 = Dense(30, activation='relu', name='bit3')(dense2)
dense4 = Dense(20, activation='relu', name='bit4')(dense3)
output1 = Dense(10, activation='relu', name='bit5')(dense4)
model1 = Model(inputs=input1, outputs=output1)
model1.summary()


x_train, x_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x_datasets, y1, y2, train_size=0.95, random_state=777,
)

last_output1 = Dense(1, name='last1')(output1)
last_output2 = Dense(1, name='last2')(output1)

model = Model(inputs = input1, outputs=[last_output1, last_output2])

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
filepath = ''.join([path1, 'k62_', date, '_', filename])


mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    verbose=1,
    save_best_only=True,
    filepath=filepath
    )

model.fit((x_train),[y1_train, y2_train], epochs=1000, validation_split=0.2, batch_size=1, callbacks=[es, mcp])

mse = model.evaluate((x_test), [y1_test, y2_test], batch_size=2)
print("mse : ", mse)

# 예측
x_predict1 = np.array([range(100, 105), range(401, 406)]).T
y_predict = model.predict(x_predict1)
print("예측 결과: ", y_predict)
loss = model.evaluate((x_test), [y1_test, y2_test])
print("로스는 ?", loss)

"""
mse :  [0.00021305083646439016, 1.735687328618951e-05, 0.00019569396681617945]
예측 결과:  [array([[3100.335 ],
       [3100.666 ],
       [3100.9985],
       [3101.3296],
       [3101.6616]], dtype=float32), array([[13101.194 ],
       [13102.377 ],
       [13103.5625],
       [13104.745 ],
       [13105.93  ]], dtype=float32)]
1/1 [==============================] - 0s 11ms/step - loss: 2.1305e-04 - last1_loss: 1.7357e-05 - last2_loss: 1.9569e-04
로스는 ? [0.00021305083646439016, 1.735687328618951e-05, 0.00019569396681617945]
로스는 ? [0.05530315637588501, 0.002880442189052701, 0.05242271348834038]
"""