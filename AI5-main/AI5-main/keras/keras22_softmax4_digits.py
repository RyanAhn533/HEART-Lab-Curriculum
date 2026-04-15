from sklearn.datasets import load_digits
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import time as t
from sklearn.metrics import r2_score
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import r2_score, accuracy_score

x, y = load_digits(return_X_y=True)
print(x)
print(y)
print(x.shape, y.shape)
print(pd.value_counts(y,sort=True))

y = pd.get_dummies(y)
print(y)

#digits - > 픽셀 사진 파일
#ascending=True : 오름차순 , sort=True 내림차순
# 3    183
# 1    182
# 5    182
# 4    181
# 6    181
# 9    180
# 7    179
# 0    178
# 2    177
# 8    174

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=3)

#모델
model = Sequential()
model.add(Dense(64, input_dim=64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))

#컴파일 훈련
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['acc'])
es= EarlyStopping(monitor='val_loss', mode = 'min', patience=300,
                  restore_best_weights=True)

model.fit(x_train, y_train, epochs=1000, batch_size=128,
          verbose=1, validation_split=0.2, callbacks=[es])

#평가예측
loss = model.evaluate(x_test, y_test)
print("로스는 ?", loss)
y_predict = model.predict(x_test)
y_pred = np.round(y_predict)
accuracy_score = accuracy_score(y_test, y_pred)
r2 = r2_score(y_test, y_predict)
print("r2스코어는? ", r2)
print("acc_score : '",accuracy_score)

