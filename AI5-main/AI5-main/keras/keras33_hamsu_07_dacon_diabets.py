from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Input
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from matplotlib import rc
from tensorflow.keras.callbacks import EarlyStopping
import time as t
import sklearn

path = "./_data/dacon/diabets/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sample_submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)
#path = "./_data/따릉/" 이렇게 이용해서 pd구문 안을 짧게 만들 수 있음

print(train_csv.columns)
print(test_csv.columns)


x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    random_state=512, train_size=0.8)
from sklearn.preprocessing import MinMaxScaler, StandardScaler,MaxAbsScaler, RobustScaler
scaler = StandardScaler()
"""
MaxAbsScaler
r2값은? -0.007563847111845545
acc_score : 0.6793893129770993
걸린 시간 : 9.79 초
r2스코어 : -0.007563847111845545

RobustScaler
r2값은? 0.07729949527346247
acc_score : 0.8091603053435115
걸린 시간 : 4.66 초
r2스코어 : 0.07729949527346247
"""
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#x_train, x_val, y_train, y_val = train_test_split(x, y, shuffle=True, random_state=5, train_size=0.2)
#모델구성
"""
model = Sequential()
model.add(Dense(16, input_dim=8, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
"""

#모델구성 2-2(함수형)
input1 = Input(shape=(8,))
dense1 = Dense(32, activation='relu', name='ys1')(input1)
dropout1 = Dropout(0.2)(dense1)
dense2 = Dense(16, activation='relu', name='ys2')(dropout1)
dropout2 = Dropout(0.2)(dense2)
dense3 = Dense(8, activation='relu', name='ys3')(dropout2)
dense4 = Dense(4, activation='relu', name='ys4')(dense3)
output1 = Dense(1, name='ys5')(dense4)

model = Model(inputs=input1, outputs=output1)

#컴파일 훈련
es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience=100, restore_best_weights=True)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
mcp = ModelCheckpoint(
    monitor='val_loss', 
    mode='auto',
    verbose=1,
    save_best_only=True, filepath=("./_save/keras32/keras32_dropout7.h1"))

model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['acc'])
start = t.time()
model.fit(x_train, y_train, epochs=100, batch_size = 8, verbose=1, 
                 validation_split=0.3, callbacks=[es])
end = t.time()


#model = load_model("./_save/keras30/keras30_6")

#평가예측
loss = model.evaluate(x_test, y_test, verbose=1)
y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('로스값은 : ', loss)
print('y값은? ', y_pred)
print('r2값은?', r2)

y_pred = np.round(y_pred)
from sklearn.metrics import r2_score, accuracy_score
accuracy_score = accuracy_score(y_test, y_pred) 
y_pred = np.round(y_pred) 
print('acc_score :', accuracy_score)
print("걸린 시간 :", round(end-start,2),'초')

y_submit = model.predict(test_csv)
print("r2스코어 :", r2)
y_submit = np.round(y_submit,2)


sample_submission_csv['Outcome'] = y_submit
sample_submission_csv.to_csv(path + "   submission_0725_9.csv")

"""
r2값은? 0.0908882961596259
acc_score : 0.7175572519083969
걸린 시간 : 4.17 초
r2스코어 : 0.0908882961596259

r2값은? 0.2428170265880274
acc_score : 0.7709923664122137
걸린 시간 : 4.51 초
r2스코어 : 0.2428170265880274
"""