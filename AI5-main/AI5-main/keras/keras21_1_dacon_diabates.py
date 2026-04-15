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
path = "./_data/dacon/diabets/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sample_submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)
#path = "./_data/따릉/" 이렇게 이용해서 pd구문 안을 짧게 만들 수 있음

print(train_csv.columns)
print(test_csv.columns)

#(652, 8) (652,)
x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    random_state=512, train_size=0.8)
#x_train, x_val, y_train, y_val = train_test_split(x, y, shuffle=True, random_state=5, train_size=0.2)
#모델구성

model = Sequential()
model.add(Dense(16, input_dim=8, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#컴파일 훈련
es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience=100, restore_best_weights=True)
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['acc'])
start = t.time()
model.fit(x_train, y_train, epochs=100, batch_size = 8, verbose=1, 
                 validation_split=0.3, callbacks=[es])
end = t.time()

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
y_submit = np.round(y_submit,2)


sample_submission_csv['Outcome'] = y_submit
sample_submission_csv.to_csv(path + "submission_0722_8.csv")
"""
r2값은? 0.11809434581836775
acc_score : 0.7099236641221374
걸린 시간 : 23.11 초
r2스코어 : 0.11809434581836775
"""