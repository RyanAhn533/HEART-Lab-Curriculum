from sklearn.datasets import load_wine
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
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

#모델2
model = Sequential()
model.add(Dense(32, input_dim=13, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))

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

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
mcp = ModelCheckpoint(
    monitor='val_loss', 
    mode='auto',
    verbose=1,
    save_best_only=True, filepath=("./_save/keras30/keras30_9_save_model.h1"))
    
model.fit(x_train,y_train, epochs=300, batch_size=8,
          verbose=1, validation_split=0.2, callbacks=[es])

model.save("./_save/keras30/keras30_9")
model = load_model("./_save/keras30/keras30_9")

#평가 예측
loss = model.evaluate(x_test, y_test)
print("로스 는? , loss")
y_predict = model.predict(x_test)
y_pred = np.round(y_predict)
accuracy_score = accuracy_score(y_test, y_pred)
r2 = r2_score(y_test, y_predict)
print("r2스코어는? ", r2)
print(y_pred)
print("acc_score : '",accuracy_score)

"""
standard
acc_score : ' 0.9629629629629629
"""