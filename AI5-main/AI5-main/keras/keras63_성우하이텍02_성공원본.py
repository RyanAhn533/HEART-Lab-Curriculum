from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,MaxPooling1D, Bidirectional,BatchNormalization, Reshape, GRU, Conv1D, Flatten, Dropout, LSTM
import sklearn as sk
from sklearn.datasets import load_boston
import numpy as np
import time as t
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input
import warnings
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import pandas as pd
path = "C:/프로그램/ai5/_data/중간고사데이터/"

naver_csv = pd.read_csv(path + "naver.csv", index_col=0, thousands=',')
naver_csv.columns = ['시가','고가','저가','종가','전일비','전일비2','등락률','거래량','금액(백만)','신용비','개인','기관','외인(수량)','외국계','프로그램','외인비']
hitech_csv = pd.read_csv(path + "hitech.csv", index_col=0, thousands=',')
hitech_csv.columns = ['시가','고가','저가','종가','전일비','전일비2','등락률','거래량','금액(백만)','신용비','개인','기관','외인(수량)','외국계','프로그램','외인비']
hybe_csv = pd.read_csv(path + "hybe.csv", index_col=0, thousands=',')
hybe_csv.columns = ['시가','고가','저가','종가','전일비','전일비2','등락률','거래량','금액(백만)','신용비','개인','기관','외인(수량)','외국계','프로그램','외인비']

naver_csv1 = naver_csv.sort_values(by=['일자'])
hitech_csv1 = hitech_csv.sort_values(by=['일자'])
hybe_csv1 = hybe_csv.sort_values(by=['일자'])

df_drop_row = naver_csv1.dropna(axis=0)
df_drop_row = hitech_csv1.dropna(axis=0)
df_drop_row = hybe_csv1.dropna(axis=0)



naver_csv3 = naver_csv1.drop(['전일비','전일비2', '프로그램', '외국계', '외인(수량)', '기관', '거래량'], axis=1)
hitech_csv3 = hitech_csv1.drop(['전일비','전일비2', '프로그램', '외국계', '외인(수량)', '기관', '거래량'], axis=1)
hybe_csv3 = hybe_csv1.drop(['전일비','전일비2', '프로그램', '외국계', '외인(수량)', '기관', '거래량'], axis=1)

x1 = naver_csv3.tail(948)
x2 = hybe_csv3.tail(948)
y1 = x1.tail(16)
y2 = x2.tail(16)
y = hitech_csv3.tail(948)

print(x1, x2, y1, y2, y)


y = y['종가']
print(y.shape)

x1 = x1.to_numpy()
x2 = x2.to_numpy()
y1 = y1.to_numpy()
y2 = y2.to_numpy()
y = y.to_numpy()


def split_x(dataset, size) :
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
        
    return np.array(aaa)
size = 16
x1 = split_x(x1, size)
x2 = split_x(x2, size)
y = split_x(y, size)


#(933, 16, 9) (933, 16, 9) (933, 16) (16, 9) (16, 9)

y1 = y1.reshape(1,16,9)
y2 = y2.reshape(1,16,9)
y = y.reshape(933,16,1)

print(x1.shape, x2.shape, y.shape)

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
    x1, x2, y, train_size=0.95, random_state=777,)


print(x1.shape, x2.shape, y1.shape, y2.shape, y.shape, x1_train.shape, x2_train.shape,y_train.shape, y_test.shape)
#(933, 16, 9) (933, 16, 9) (1, 16, 9) (1, 16, 9) (933, 16, 1) (886, 16, 9) (886, 16, 9) (886, 16, 1) (47, 16, 1)

#2-1 모델
input1 = Input(shape=(16,9))
dense1 = Bidirectional(LSTM(32, return_sequences=True, name='bit1'))(input1)
reshape2 = Reshape(target_shape=(16,64))(dense1)
dense3 = Conv1D(64, kernel_size=2, padding="causal")(reshape2)
maxpool1 = MaxPooling1D(pool_size=2)(dense3)
dropout4 = Dropout(0.2)(maxpool1)
batch1 = BatchNormalization()(dropout4)
dense5 = Dense(128, activation='relu')(batch1)
flatten1 = Flatten()(dense5)
# flatten1 = flatten1 
dense6 = Dense(64, activation='relu')(flatten1)
output1 = Dense(32, activation='relu')(dense6)
model1 = Model(inputs=input1, outputs=output1)

model1.summary()

#모델2_2
input11 = Input(shape=(16,9))
dense11 = Bidirectional(LSTM(32, return_sequences=True))(input11)
reshape2 = Reshape(target_shape=(16,64))(dense11)
dense12 = Conv1D(64, kernel_size=2, padding="causal")(reshape2)
maxpool2 = MaxPooling1D(pool_size=2)(dense12)
dropout3 = Dropout(0.2)(maxpool2)
batch2 = BatchNormalization()(dropout3)
dense13 = Dense(128, activation='relu')(dropout3)
flatten2 = Flatten()(dense13) 
# flatten2 = flatten2
dense14 = Dense(64, activation='relu')(flatten2)
dense15 = Dense(32, activation='relu')(dense14)
output11 = Dense(16, activation='relu')(dense15)
model11 = Model(inputs=input11, outputs=output11)
model11.summary()



#3. 합체!!!
from keras.layers.merge import concatenate, Concatenate

merge1 = Concatenate(name='mg1')([output1, output11])
last_output = Dense(16, name='last')(merge1)

model = Model(inputs = [input1, input11], outputs=last_output)
model.summary()

#3. 컴파일, 훈련
import time
model.compile(loss='mse', optimizer='adam', metrics = ['mse'])
start = time.time()
es = EarlyStopping(
    monitor = 'loss',
    mode = 'min',
    verbose=1,
    patience=50,
    restore_best_weights=True
)
import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')

path1 = 'C:\\프로그램\\ai5\\_save\\중간고사가중치\\'
filename = '성우하이텍_안준영.hdf5'
filepath = ''.join([path1, 'save_model', filename])


mcp = ModelCheckpoint(
    monitor= 'val_loss',
    mode = 'auto',
    verbose=1,
    save_best_only= True,
    filepath = filepath
)
model.fit([x1_train, x2_train],y_train, epochs=1000, validation_split=0.2, callbacks=[es,mcp], batch_size=128)

end = time.time()

mse = model.evaluate([x1_test, x2_test], y_test, batch_size=128)


# 예측
# x_predict1 = np.array(x_1_predict)
# x_predict2 = np.array(x_2_predict)
# y_predict = model.predict([x_predict1, x_predict2])
# print("예측 결과: ", y_predict)
loss = model.evaluate([x1_test,x2_test], y_test)
print("로스는 ?", loss)

y_pred = model.predict([y1, y2])
print("예측 결과: ", y_pred[0:1,0])
print("mse : ", mse)

"""
로스는 ? [1965584.0, 1965584.0]
예측 결과:  [[5812.8237 5749.939  5732.703  5742.4575 5712.0977 5732.102  5761.873
  5809.9443 5812.121  5888.9624 5793.7236 5783.3325 5803.4653 5999.896
  5888.593  5902.9165]]
mse :  [1965584.0, 1965584.0]
"""