#url : https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data?select=test.csv
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Input
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats

path = "C:/프로그램/ai5/_data/kaggle/santander/"

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
sampleSubmission = pd.read_csv(path + 'sample_submission.csv', index_col=0)

x = train_csv.drop('target', axis=1)
y = train_csv['target']

"""
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
"""

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=3, stratify=y)
from sklearn.preprocessing import MinMaxScaler, StandardScaler,MaxAbsScaler, RobustScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
test_csv = scaler.fit_transform(test_csv)

"""
MaxAbsScaler
loss : 0.5624394416809082
acc : 0.789
r2_score :  0.6154876429320677
acc_score :  0.7407886231415644

RobustScaler
loss : 0.5512962341308594
acc : 0.792
r2_score :  0.6248450659451026
acc_score :  0.7491919844861021
"""

print(x_train.shape)
print(y_train.shape)


#모델
input1 = Input(shape=(200,))
dense1 = Dense(64,activation='relu', name='ys1')(input1)
dropout1 = Dropout(0.2)(dense1)
dense2 = Dense(16, activation='relu', name='ys2')(dropout1)
dropout2 = Dropout(0.2)(dense2)
dense3 = Dense(8, activation='relu', name='ys3')(dropout2)
dense4 = Dense(4, activation='relu', name='ys4')(dense3)
output1 = Dense(1, activation='softmax', name='ys5')(dense4)
model = Model(inputs=input1, outputs=output1)

#컴파일 훈련
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['acc'])
es= EarlyStopping(monitor='val_loss', mode = 'min', patience=20,
                  restore_best_weights=True)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
mcp = ModelCheckpoint(
    monitor='val_loss', 
    mode='auto',
    verbose=1,
    save_best_only=True, filepath=("./_save/keras31_mcp/keras31_mcp_12_santander.h1"))

model.fit(x_train, y_train, epochs=50, batch_size=128,
          verbose=1, validation_split=0.2, callbacks=[es])



#평가예측

loss = model.evaluate(x_test, y_test)
print('loss :', loss[0])
print('acc :', round(loss[1],3))

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2_score : ', r2)
y_pred = np.round(y_pred)
accuracy_score = accuracy_score(y_test, y_pred)
print('acc_score : ', accuracy_score)



y_submit = model.predict(test_csv)
y_submit = np.round(y_submit)

print(y_submit[:10])

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)

print("걸린시간은?", "gpu on" if (len(gpus) > 0) else "gpu off", round(end - start, 2), '초')
if(gpus):
    print("쥐피유 돈다!!!")
else:
    print("쥐피유 없다!")