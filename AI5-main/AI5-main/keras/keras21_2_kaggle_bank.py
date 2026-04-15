#https://www.kaggle.com/competitions/playground-series-s4e1/data?select=train.csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from matplotlib import rc
from tensorflow.keras.callbacks import EarlyStopping
import time as t
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
path = "./_data/kaggle/Bank/"

# 0 replace data
train_csv = pd.read_csv(path + "train.csv", index_col = 0)
test_csv = pd.read_csv(path + "test.csv", index_col = 0)

print(train_csv['Geography'].value_counts())
"""
train_csv['Geography'] = train_csv['Geography'].replace('France', value = 1)
train_csv['Geography'] = train_csv['Geography'].replace('Spain', value = 2)
train_csv['Geography'] = train_csv['Geography'].replace('Germany', value = 3)

test_csv['Geography'] = test_csv['Geography'].replace('France', value = 1)
test_csv['Geography'] = test_csv['Geography'].replace('Spain', value = 2)
test_csv['Geography'] = test_csv['Geography'].replace('Germany', value = 3)

train_csv['Gender'] = train_csv['Gender'].replace('Male', value = 1)
train_csv['Gender'] = train_csv['Gender'].replace('Female', value = 2)

train_csv.to_csv(path + "replaced_train.csv")

test_csv['Gender'] = test_csv['Gender'].replace('Male', value = 1)
test_csv['Gender'] = test_csv['Gender'].replace('Female', value = 2)

test_csv.to_csv(path + "replaced_test.csv")

"""
re_train_csv = pd.read_csv(path + "replaced_train.csv", index_col=0)
re_test_csv = pd.read_csv(path + "replaced_test.csv", index_col=0)



re_train_csv.info()
re_test_csv.info()

sample_submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)
 
print(re_train_csv.columns)
print(re_test_csv.columns)

re_test_csv = re_test_csv.drop(['CustomerId'], axis=1)

x = re_train_csv.drop(['Exited','CustomerId'], axis=1)
y = re_train_csv['Exited']
print(type(x)) #
print(type(y)) #165034 rows x 12 columns]
print(re_train_csv.isna().sum())   # 0
print(re_test_csv.isna().sum())    # 0

############################
train_scaler = MinMaxScaler()

train_csv_copy = re_train_csv.copy()

train_scaler.fit(train_csv_copy)

train_csv_scaled = train_scaler.transform(train_csv_copy)

re_train_csv = pd.concat([pd.DataFrame(data = train_csv_scaled), re_train_csv['Exited']], axis = 1)

re_test_scaler = MinMaxScaler()

test_csv_copy = re_test_csv.copy()

re_test_scaler.fit(test_csv_copy)

re_test_csv_scaled = re_test_scaler.transform(test_csv_copy)

test_csv = pd.DataFrame(data = re_test_csv_scaled)

####################
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    random_state=512, train_size=0.8)
#x_train, x_val, y_train, y_val = train_test_split(x, y, shuffle=True, random_state=5, train_size=0.2)
#모델구성

model = Sequential()
model.add(Dense(16, input_dim=10, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#컴파일 훈련
es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience=100, restore_best_weights=True)
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['acc'])
start = t.time()
model.fit(x_train, y_train, epochs=1000, batch_size = 128, verbose=1, 
                 validation_split=0.3, callbacks=[es])
end = t.time()

#평가예측
loss = model.evaluate(x_test, y_test, verbose=1)
y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
y_pred = np.round(y_pred)

accuracy_score = accuracy_score(y_test, y_pred) 
y_pred = np.round(y_pred) 
y_submit = model.predict(re_test_csv)
y_submit = np.round(y_submit)
print('로스값은 : ', loss)
print('y값은? ', y_pred)
print("r2스코어 :", r2)
print('acc_score :', accuracy_score)
print("걸린 시간 :", round(end-start,2),'초')



sample_submission_csv['Exited'] = y_submit
sample_submission_csv.to_csv(path + "submission_0723_gpt.csv")
