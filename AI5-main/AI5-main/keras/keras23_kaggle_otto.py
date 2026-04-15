#url : https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data?select=test.csv
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler


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

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, 
                                                    random_state=3, stratify=y)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)






#모델
model = Sequential()
model.add(Dense(512, input_dim=93, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(9, activation='softmax'))

#컴파일 훈련
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['acc'])
es= EarlyStopping(monitor='val_loss', mode = 'min', patience=2,
                  restore_best_weights=True)

model.fit(x_train, y_train, epochs=100, batch_size=1024,
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

"""
samplesubmission_csv['calss1'] = y_submit[:0].astype('int')
samplesubmission_csv['calss2'] = y_submit[:1].astype('int')
samplesubmission_csv['calss3'] = y_submit[:2].astype('int')
samplesubmission_csv['calss4'] = y_submit[:3].astype('int')
samplesubmission_csv['calss5'] = y_submit[:4].astype('int')
samplesubmission_csv['calss6'] = y_submit[:5].astype('int')
samplesubmission_csv['calss7'] = y_submit[:6].astype('int')
samplesubmission_csv['calss8'] = y_submit[:7].astype('int')
samplesubmission_csv['calss9'] = y_submit[:8].astype('int')
"""



for i in range(9) :
    samplesubmission1_csv['Class_' + str(i + 1)] = y_submit[:, i].astype('int')

samplesubmission1_csv.to_csv(path + "otto_lotto_1.csv")






"""
#평가예측
loss = model.evaluate(x_test, y_test)
print("로스는 ?", loss)
y_predict = model.predict(x_test)
y_pred = np.round(y_predict)
accuracy_score = accuracy_score(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("r2스코어는? ", r2)
print("acc_score : '",accuracy_score)

exit()
"""