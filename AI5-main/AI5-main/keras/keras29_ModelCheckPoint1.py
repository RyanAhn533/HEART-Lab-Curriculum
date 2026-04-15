from sklearn.datasets import load_boston   
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
from tensorflow.keras.models import Sequential, load_model
#1. 데이터 
dataset = load_boston()

x = dataset.data    
y = dataset.target  


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=231)

from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train)
print(np.min(x_train), np.max(x_train))    
print(x_test)
print(np.min(x_test), np.max(x_test))     


#2. 모델 구성
model = Sequential()
model.add(Dense(32, input_dim=13, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=10, restore_best_weights=True)
mcp = ModelCheckpoint(
    monitor='val_loss', 
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath='./_save/keras29_mcp/keras29_mcp1.hdf5'
)
start = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=16,
          verbose=1, 
          validation_split=0.1, callbacks=[es, mcp]
          )
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)    # 추가
print('loss :', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 score :', r2)


#모델체크
