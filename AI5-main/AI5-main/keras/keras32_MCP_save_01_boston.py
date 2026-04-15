from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import sklearn as sk
from sklearn.datasets import load_boston
import numpy as np
import time as t
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.callbacks import EarlyStopping

dataset = load_boston()
print(dataset)
print(dataset.DESCR)
print(dataset.feature_names)
#['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
# 'B' 'LSTAT']



x = dataset.data
y = dataset.target


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=3)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train)
print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))



#모델
model = Sequential()
model.add(Dense(128, input_dim=13, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()


#model = load_model("./_save/keras28/keras28_1_save_model.h5")

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start = t.time()
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor='val_loss', mode = 'min', #모르면 auto / min max auto
    patience=10, 
    restore_best_weights=True
)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
mcp = ModelCheckpoint(
    monitor='val_loss', 
    mode='auto',
    verbose=1,
    save_best_only=True, filepath=("./_save/keras30/keras30_1_save_model.h1"))

hist = model.fit(x_train, y_train, epochs=1000, verbose = 2, 
                 batch_size=32, validation_split=0.3,
                 callbacks=[es, mcp]) # valtidation_data=(x_val, y_val)
end  = t.time()



#평가, 예측
loss = model.evaluate(x_test, y_test)
print("로스 :", loss)

y_predict = model.predict(x_test)


r2 = r2_score(y_test, y_predict) 