from sklearn.datasets import load_wine
import numpy as np
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Input
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import time as t
from sklearn.metrics import r2_score
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import r2_score, accuracy_score

#1.데이터
datasets = load_wine()
x = datasets.data
y = datasets.target
y = pd.get_dummies(y)

print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=138, shuffle=True)
from sklearn.preprocessing import MinMaxScaler, StandardScaler,MaxAbsScaler, RobustScaler
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

"""
MaxAbsScaler
acc_score : ' 0.9629629629629629
RobustScaler
acc_score : ' 1.0
"""
"""
#모델2
model = Sequential()
model.add(Dense(32, input_dim=13, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
"""
#모델2-2(함수형)
input1 = Input(shape=(13,))
dense1 = Dense(32, activation='relu', name='ys1')(input1)
dropout1 = Dropout(0.2)(dense1)
dense2 = Dense(16, activation='relu', name='ys2')(dropout1)
dropout2 = Dropout(0.2)(dense2)
dense3 = Dense(8, activation='relu', name='ys3')(dropout2)
dense4 = Dense(4, activation='relu', name='ys4')(dense3)
output1 = Dense(3, name='ys5')(dense4)

model = Model(inputs=input1, outputs=output1)

#컴파일, 훈련
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss',
    mode = 'min',
    patience=100,
    restore_best_weights=True
)
import time
start = time.time()
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
mcp = ModelCheckpoint(
    monitor='val_loss', 
    mode='auto',
    verbose=1,
    save_best_only=True, filepath=("./_save/keras32/keras32.Dropout. wine9.h1"))

model.fit(x_train,y_train, epochs=300, batch_size=8,
          verbose=1, validation_split=0.2, callbacks=[es])
end = time.time()

#평가 예측
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)
y_pred = np.round(y_predict)
accuracy_score = accuracy_score(y_test, y_pred)
r2 = r2_score(y_test, y_predict)
print("r2스코어는? ", r2)
print(y_pred)
print("acc_score : '",accuracy_score)
print("로스 는? ", loss)

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)

print("걸린시간은?", "gpu on" if (len(gpus) > 0) else "gpu off", round(end - start, 2), '초')
if(gpus):
    print("쥐피유 돈다!!!")
else:
    print("쥐피유 없다!")
    
"""
acc_score : ' 1.0
로스 는?  [1.9028821043320931e-06, 1.0]

acc_score : ' 0.9814814814814815
로스 는?  [0.1777685135602951, 0.9814814925193787]
"""