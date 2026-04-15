#https://www.kaggle.com/datasets/maciejgronczynski/biggest-genderface-recognition-dataset/data

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, BatchNormalization
import time
from sklearn.model_selection import train_test_split

start_time=time.time()
np_path = 'c:/프로그램/ai5/_data/kaggle/biggest_gender/'
x_train1=np.load(np_path + 'keras45_07_x_train1.npy')
y_train1=np.load(np_path + 'keras45_07_y_train1.npy')
x_train2=np.load(np_path + 'keras45_07_x_train2.npy')
y_train2=np.load(np_path + 'keras45_07_y_train2.npy')

x_train= np.concatenate((x_train1[:5000],x_train2))
y_train= np.concatenate((y_train1[:5000],y_train2))

end_time=time.time()
print("데이터 불러오는 시간 :", round(end_time-start_time,2),'초') 

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.9, random_state=5656)

# #2. modeling
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 3), padding='same')) 
model.add(MaxPool2D())
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(filters=64, activation='relu', kernel_size=(3,3), padding='same')) 
model.add(MaxPool2D())
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(filters=128, activation='relu', kernel_size=(3,3), padding='same')) 
model.add(MaxPool2D())
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), activation='relu', padding='same')) 
model.add(MaxPool2D())
model.add(Dropout(0.25))

model.add(Flatten()) 
model.add(Dense(1024, activation='relu')) 
model.add(Dense(512, activation='relu')) 
model.add(Dense(1, activation='sigmoid'))
                                          
                        
#3. compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'acc', 'mse'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=30, verbose=1, restore_best_weights=True)

################## mcp 세이브 파일명 만들기 시작 ###################
import datetime
date = datetime.datetime.now()
print(date) #2024-07-26 16:49:57.565880
print(type(date)) #<class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")
print(date) #0726_1654
print(type(date)) #<class 'str'>


path = 'C:\\ai5\\_save\\keras45\\k45_07\\'
filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
filepath = "".join([path, 'k45_07_', date, '_' , filename])
#생성 예 : ./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5
################## mcp 세이브 파일명 만들기 끝 ################### 

mcp=ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose = 1,
    save_best_only=True,
    filepath=filepath)


start_time=time.time()
model.fit(x_train, y_train, epochs=30, batch_size=16, validation_split=0.2, callbacks=[es, mcp])
# model.fit_generator(x_train, y_train,
#                     epochs=1000,
#                     verbose=1,
#                     callbacks=[es, mcp],
#                     validation_steps=50)
end_time=time.time()


#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', round(loss[1],5))

y_pre = np.round(model.predict(x_test, batch_size=16))
print("걸린 시간 :", round(end_time-start_time,2),'초')

# y_pre = np.round(y_pre)

### csv 파일 만들기 ###
# y_pre = np.clip(y_pre, 1e-6, 1-(1e-6))
# print(y_submit)

# y_submit = np.round(y_submit)
# print(y_submit)

# print(y_submit)
# sample_submission['label'] = y_submit
# sample_submission.to_csv(path_submission + "sampleSubmission_0803_11.csv")

"""
loss : 0.345674604177475
acc : 0.91833
걸린 시간 : 266.96 초

loss : 0.345674604177475
acc : 0.91833
걸린 시간 : 266.96 초
"""