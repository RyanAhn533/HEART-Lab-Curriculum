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

path = "C:/프로그램/ai5/_data/kaggle/otto/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
samplesubmission1_csv = pd.read_csv(path + "samplesubmission.csv", index_col=0)

print(train_csv.select_dtypes(include=['object']).columns)
print(test_csv.select_dtypes(include=['object']).columns)

train_csv.info()
test_csv.info()
print(train_csv['target'].value_counts())
train_csv['target'] = train_csv['target'].replace({'Class_1' : 1, 'Class_1' : 1, 'Class_2' : 2, 'Class_3' : 3, 'Class_4' : 4, 'Class_5' : 5, 'Class_6' : 6, 'Class_7' : 7, 'Class_8' : 8, 'Class_9' : 9, })



x = train_csv.drop(['target'], axis=1)
"""
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
"""
y = train_csv['target']
print(x.shape)
print(y.shape)
y = pd.get_dummies(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=3, stratify=y)

from sklearn.preprocessing import MinMaxScaler, StandardScaler,MaxAbsScaler, RobustScaler
scaler = RobustScaler()
"""
MaxAbsScaler
loss : 0.6057980060577393
acc : 0.775
r2_score :  0.5894159728629274
acc_score :  0.7240896358543417

RobustScaler
loss : 0.5655195116996765
acc : 0.791
r2_score :  0.6151580478947759
acc_score :  0.7378258995906055
"""

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)



"""

#모델
model = Sequential()
model.add(Dense(512, input_dim=93, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(9, activation='softmax'))


"""

input1 = Input(shape=(93,))
dense1 = Dense(10, activation='relu', name='ys1')(input1)
dropout1 = Dropout(0.2)(dense1)
dense2 = Dense(16, activation='relu', name='ys2')(dropout1)
dropout2 = Dropout(0.2)(dense2)
dense3 = Dense(8, activation='relu', name='ys3')(dropout2)
dense4 = Dense(4, activation='relu', name='ys4')(dense3)
output1 = Dense(9, activation='softmax', name='ys5')(dense4)

model = Model(inputs=input1, outputs=output1)

#컴파일 훈련
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['acc'])
es= EarlyStopping(monitor='val_loss', mode = 'min', patience=2,
                  restore_best_weights=True)
start = time.time()
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
mcp = ModelCheckpoint(
    monitor='val_loss', 
    mode='auto',
    verbose=1,
    save_best_only=True, filepath=("./_save/keras32/keras32_13_save_model.h1"))

model.fit(x_train, y_train, epochs=100, batch_size=1024,
          verbose=1, validation_split=0.2, callbacks=[es])

end = time.time()

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
y_submit = np.round(y_submit,1)

print(y_submit[:10])



import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)

print("걸린시간은?", "gpu on" if (len(gpus) > 0) else "gpu off", round(end - start, 2), '초')
if(gpus):
    print("쥐피유 돈다!!!")
else:
    print("쥐피유 없다!")
    

"""
for i in range(9) :
    samplesubmission1_csv['Class_' + str(i + 1)] = y_submit[:, i].astype('int')

samplesubmission1_csv.to_csv(path + "otto_lotto_2.csv")
"""
