import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

# 데이터 불러오기
path = "C:/프로그램/ai5/_data/중간고사데이터/"
naver_csv = pd.read_csv(path + "naver.csv", index_col=0, names=(['일자','시가','고가','저가','종가','전일비','전일비2','등락률','거래량','금액(백만)','신용비','개인','기관','외인(수량)','외국계','프로그램','외인비']))
hitech_csv = pd.read_csv(path + "hitech.csv", index_col=0, names=(['일자','시가','고가','저가','종가','전일비','전일비2','등락률','거래량','금액(백만)','신용비','개인','기관','외인(수량)','외국계','프로그램','외인비']))
hybe_csv = pd.read_csv(path + "hybe.csv", index_col=0, names=(['일자','시가','고가','저가','종가','전일비','전일비2','등락률','거래량','금액(백만)','신용비','개인','기관','외인(수량)','외국계','프로그램','외인비']))

# 전처리
naver_csv = naver_csv.drop(['전일비','전일비2'], axis=1).sort_values(by=['일자'])
hitech_csv = hitech_csv.drop(['전일비','전일비2'], axis=1).sort_values(by=['일자'])
hybe_csv = hybe_csv.drop(['전일비','전일비2'], axis=1).sort_values(by=['일자'])

# 필요한 데이터 추출
x1 = naver_csv['종가'].to_numpy()[-948:-16]
x2 = hitech_csv['종가'].to_numpy()[-948:-16]
y = hybe_csv['종가'].to_numpy()[-948:-16]

# 데이터 분할 함수 정의
def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i:(i + size)]
        aaa.append(subset)
    return np.array(aaa)

size = 16
x1 = split_x(x1, size)
x2 = split_x(x2, size)
y = split_x(y, size)

# 데이터를 3차원 배열로 변환
x1 = x1.reshape((x1.shape[0], x1.shape[1], 1))
x2 = x2.reshape((x2.shape[0], x2.shape[1], 1))
y = y[:, -1]  # y의 마지막 값을 예측 대상으로 설정

# Train-Test Split
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, train_size=0.95, random_state=777)

# 모델1
input1 = Input(shape=(x1_train.shape[1], 1))
lstm1 = LSTM(8, activation='relu')(input1)
dense1 = Dense(16, activation='relu')(lstm1)
dense2 = Dense(32, activation='relu')(dense1)
output1 = Dense(16, activation='relu')(dense2)

# 모델2
input2 = Input(shape=(x2_train.shape[1], 1))
lstm2 = LSTM(8, activation='relu')(input2)
dense3 = Dense(16, activation='relu')(lstm2)
dense4 = Dense(32, activation='relu')(dense3)
output2 = Dense(16, activation='relu')(dense4)

# 모델 합치기
merge = Concatenate()([output1, output2])
merge_dense = Dense(16, activation='relu')(merge)
last_output = Dense(1)(merge_dense)

model = Model(inputs=[input1, input2], outputs=last_output)

# 컴파일 및 학습
model.compile(loss='mse', optimizer='adam')

# 콜백 설정
es = EarlyStopping(monitor='val_loss', mode='min', patience=5, restore_best_weights=True)
mcp = ModelCheckpoint(filepath='model.h5', save_best_only=True, monitor='val_loss', mode='min')

# 모델 학습
model.fit([x1_train, x2_train], y_train, epochs=100, batch_size=16, validation_split=0.2, callbacks=[es, mcp])

# 평가 및 예측
mse = model.evaluate([x1_test, x2_test], y_test)
print("MSE: ", mse)

y_pred = model.predict([x1_test, x2_test])
print("예측 결과: ", y_pred)
