import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, LSTM
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
'''
{'참': 1, '너무': 2, '재미있다': 3, '최고에요': 4, '잘만든': 5,
 '영화예요': 6, '추천하고': 7, '싶은': 8, '영화입니다': 9, 
 '한': 10, '번': 11, '더': 12, '보고': 13, '싶어요': 14, '글쎄': 15,
 '별로에요': 16, '생각보다': 17, '지루해요': 18, '연기가': 19, '어색해요': 20,
 '재미없어요': 21, '재미없다': 22, '재밋네요': 23, '준영이': 24, '바보': 25, 
 '반장': 26, '잘생겼다': 27, '태운이': 28, '또': 29, '구라친다': 30}
 '''
x = token.texts_to_sequences(docs)
x_predict = token.texts_to_sequences(docs2)

print(x)
''' # <class 'lst>
[[2, 3], [1, 4], [1, 5, 6], [7, 8, 9], [10, 11, 12, 13, 14],
 [15], [16], [17, 18], [19, 20], [21], [2, 22], [1, 23],
 [24, 25], [26, 27], [28, 29, 30]]'''
print(type(x))


#길이가 맞지 않기 때문에 가장 긴 값을 기준으로 0을 넣어준다
#ex) [0, 0, 0, 2, 3]
"""
max_len = max(len(item) for item in x)
print('최대 길이 :',max_len)


for sentence in x:
    while len(sentence) < max_len:
        sentence.append(0)

padded_np = np.array(x)


x1 = token.texts_to_sequences(x)
print(x1)
x_padded = pad_sequences(x1)

from tensorflow.keras.preprocessing.sequence import pad_sequences


x = pd.DataFrame(x)
ohe = OneHotEncoder(sparse=False)
x_encoded2 = np.array(x).reshape(-1,1) 
x = ohe.fit_transform(x_encoded2)

"""


from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x)#padding='pre', 'post', maxlen=5, truncating='pre', 'post')
x_pred = pad_sequences(x_predict, maxlen=5)#padding='pre', 'post', maxlen=5, truncating='pre', 'post')

print(pad_x)
print(pad_x.shape)
print(x_pred.shape)

x_train, x_test, y_train, y_test = train_test_split(pad_x, labels, train_size=0.8, random_state=5656)


model = Sequential()
model.add(Dense(10, input_shape=(5,))) # timesteps , features
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
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
model.fit(x_train,y_train, epochs=100, batch_size=64, verbose=1)

loss= model.evaluate(x_test, y_test)
y_predict = np.round(model.predict(x_pred))
print('로스 값 : ', loss[0])
print(y_predict)