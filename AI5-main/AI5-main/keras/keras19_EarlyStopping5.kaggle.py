# 기존 캐글 데이터에서

# 1. train)csv의 y를 casual과 register로 잡는다.
# 이후 훈련을 해서 test_csv의 casual과 register를 predict한다
# 2. test_csv에 casual과 register 컬럼을 합쳐서
# 3. train_csv에 y를 count로 잡는다.
# 4. 전체 훈련
# 5. test_csv 예측해서 submission에 붙여!!

#환경구축
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time as t
from tensorflow.keras.callbacks import EarlyStopping



# 1. 데이터
path = 'C:\\프로그램\\ai5\\_data/kaggle/bike-sharing-demand\\'
# \a  \b 이런걸 하나의 문자로 인식함 줄바꿈 이런거
# # 위와같은 애들 \ -> \\로 바꿔줘야함 / // 도 가능
# path = 'C:/프로그램//ai5\_data\\bike-sharing-demand'

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test2_csv = pd.read_csv(path + "test2.csv", index_col=0)
sampleSubmission_csv = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

print(train_csv.shape)
print(test2_csv.shape)
print(sampleSubmission_csv.shape)

x = train_csv.drop(['count'], axis = 1)
y = train_csv['count'] #, 'registered'

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=433, train_size=0.9)

#모델구성
model = Sequential()
model.add(Dense(16,activation='relu', input_dim=10))
model.add(Dense(32,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1, activation='linear'))


#컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
es = EarlyStopping(monitor = 'val_loss', mode ='min', patience=20, restore_best_weights=True)
start_tiem = t.time()
hist = model.fit(x_train, y_train, epochs=500, batch_size=762, 
                 verbose=2, validation_split=0.2, callbacks=[es])
end_time = t.time()
#결과
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)
#print(y_predict)
y_submit = model.predict(test2_csv)
sampleSubmission_csv['count'] = y_submit
sampleSubmission_csv.to_csv(path + "submission_0718_1423.csv")


import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(9,6))        #그림판의 사이즈를 9 by6으로
plt.plot(hist.history['loss'], color='red', label='loss')
plt.plot(hist.history['val_loss'], color='blue', label='val_loss')
plt.legend(loc='upper right')
plt.title('케긁바이크')
plt.xlabel('epochs')
plt.xlabel('loss')
plt.grid()
plt.show()
print("###########걸린시간##############")
print('걸리는 시간은?', round(end_time-start_time, 1), "초")

