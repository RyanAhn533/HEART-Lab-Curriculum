import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, LSTM, Conv1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 데이터 경로 설정
path = "C:/프로그램/ai5/_data/kaggle/santander/"

# 데이터 로드
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
sampleSubmission = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# 데이터 분리
x = train_csv.drop('target', axis=1)
y = train_csv['target']

# 데이터 스케일링
scaler = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=3, stratify=y)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

# 데이터 형태 변환
x_train = x_train.reshape(x_train.shape[0], 200, 1)
x_test = x_test.reshape(x_test.shape[0], 200, 1)

'''
# 모델 생성
model = Sequential()
model.add(Dense(64, input_dim=200, activation='relu')) 
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(16, activation='relu')) 
model.add(Dense(1, activation='sigmoid'))  # 이진 분류에 맞게 수정


model = Sequential()
model.add(LSTM(10, input_shape=(200, 1))) # timesteps , features
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(1))
'''
model = Sequential()
model.add(Conv1D(filters=10, kernel_size=2, input_shape=(200, 1)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(2048, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(1))
model.summary()
# 모델 컴파일 및 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
es= EarlyStopping(monitor='val_loss', mode = 'min', patience=20, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=("./_save/keras31_mcp/keras31_mcp_12_santander.h5"))

model.fit(x_train, y_train, epochs=50, batch_size=128, verbose=1, validation_split=0.2, callbacks=[es])

# 평가 및 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss[0])
print('acc :', round(loss[1],3))

y_pred = model.predict(x_test)
y_pred = np.round(y_pred)
accuracy_score_value = accuracy_score(y_test, y_pred)
print('acc_score : ', accuracy_score_value)
'''
loss : 0.24196647107601166
acc : 0.91
acc_score :  0.91005
후

loss : 1.5499556064605713
acc : 0.9
acc_score :  0.7211166666666666

로스는 ? 0.8074327111244202
r2스코어는?  0.9031190259018818
'''
