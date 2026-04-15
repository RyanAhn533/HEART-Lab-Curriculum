import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling2D, LSTM
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

#1. 데이터
docs = ['너무 재미있다', '참 최고에요', '참 잘만든 영화예요',
        '추천하고 싶은 영화입니다', '한 번 더 보고 싶어요', '글쎄',
        '별로에요', '생각보다 지루해요', '연기가 어색해요',
        '재미없어요', '너무 재미없다.', '참 재밋네요.',
        '준영이 바보', '반장 잘생겼다', '태운이 또 구라친다']

labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0])
docs2 = ["태운이 참 재미없다."]
token = Tokenizer()
token.fit_on_texts(docs)
token.fit_on_texts(docs2)
print(token.word_index)

x = token.texts_to_sequences(docs)
x_predict = token.texts_to_sequences(docs2)

print(x_predict)

''' # <class 'lst>
[[2, 3], [1, 4], [1, 5, 6], [7, 8, 9], [10, 11, 12, 13, 14],
 [15], [16], [17, 18], [19, 20], [21], [2, 22], [1, 23],
 [24, 25], [26, 27], [28, 29, 30]]'''
print(type(x))


#길이가 맞지 않기 때문에 가장 긴 값을 기준으로 0을 넣어준다
#ex) [0, 0, 0, 2, 3]



from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x)#padding='pre', 'post', maxlen=5, truncating='pre', 'post')
x_pred = pad_sequences(x_predict, maxlen=5)#padding='pre', 'post', maxlen=5, truncating='pre', 'post')
print(x_pred.shape)

print(pad_x)
print(pad_x.shape)
print(x_pred.shape)
print(type(pad_x))
print(x_pred) 

#(15, 5)
#(1, 5)
x_predict = to_categorical(x_pred, num_classes=31)
# ohe = OneHotEncoder(sparse=False)
# x_predict = ohe.fit_transform(x_pred)

#(1, 31,5)
ohe = OneHotEncoder(sparse=False)
x = np.array(pad_x).reshape(-1,1)
ohe.fit(x)
x_ = ohe.transform(x)
print(x_.shape)
print(x_predict.shape)

x = x_.reshape(15,5,31)

x_train, x_test, y_train, y_test = train_test_split(x, labels, train_size=0.8, random_state=5656)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
#(12, 5) (3, 5) (12,) (3,)

model = Sequential()
model.add(Conv1D(64, kernel_size=2, input_shape=(5,31)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])

import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')

path1 = './_save/keras65/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path1, 'k65_', date, '_', filename])

es = EarlyStopping(
    monitor='val_loss',
    mode = 'min',
    patience=30,
    verbose=1,
    restore_best_weights=True
)

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    save_best_only = True,
    verbose=1,
    filepath=filepath
)
model.fit(x_train,y_train, epochs=10000, batch_size=2, verbose=1)

loss= model.evaluate(x_test, y_test)
y_predict = np.round(model.predict(x_predict))
print('로스 값 : ', loss[0])
print(y_predict)