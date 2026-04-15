#65_4 카피
# 원핫인코딩을 하는 이유는 인덱스들이 혼합되면서 혼동되는 문제를 해결한다. 
# 하지만, 데이터의 수가 늘어나면 인덱스가 기하급수적으로 늘어나는 문제가 발생한다.
# 또한, 0의 갯수가 엄청 늘어나면서 연산량도 늘어나고, 결과값이 0으로 수렴한다.

# 이러한 문제를 해결하기 위해서 의미가 유사한 것을 기반으로 군집화할 수 있다. = Embedding


from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, Flatten

#1. 데이터
docs = [
    '너무 재미있다', '참 최고에요', '참 잘만든 영화예요',
    '추천하고 싶은 영화입니다', '한 번 더 보고 싶어요', '글쎄',
    '별로에요', '생각보다 지루해요', '연기가 어색해요', 
    '재미없어요', '너무 재미없다', '참 재밋네요', 
    '준영이 바보', '반장 잘생겼다', '태운이 또 구라친다'
]

labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0])
y = ["태운이 참 재미없다."]

token = Tokenizer()
token.fit_on_texts(docs)

x = token.texts_to_sequences(docs)
y = token.texts_to_sequences(y)
print(token.word_index)

#### pad_sequence ###
from keras.preprocessing.sequence import pad_sequences
x = pad_sequences(x, maxlen = 5)
y = pad_sequences(y, maxlen = 5)

# print(x.shape, y.shape) #(15, 5) (1, 5)
# 임베딩을 하면서 데이터의 가로세로는 맞춰주어야 하기 때문에, pad_sequences를 마친 데이터를 입력


#2. 모델
from keras.layers import Embedding
model = Sequential()
# -----------------------임베딩1------------------------------------------
# model.add(Embedding(input_dim=31, # 단어 사전의 갯수
#                     output_dim=100, # 다음에 전달하는 노드의 수
#                     input_length=5  #(15, 5) --> (None, 5, 100)출력
#                     ))
#  Layer (type)                Output Shape              Param #
# =================================================================
#  embedding (Embedding)       (None, 5, 100)            3100
# =================================================================
# Total params: 3,100

# model.summary() #embedding (Embedding)(None, 5, 100)3100  // 2차원 data형태로 입력하여 3차원 data의 출력
# # 이로 인해, 자연어 처리와 시계열 데이터에서 활용




# -----------------------임베딩2------------------------------------------
# model.add(Embedding(input_dim=31, output_dim=100)) # input_length를 지정하지 않아도 자동으로 맞춰줌
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  embedding (Embedding)       (None, None, 100)         3100
#  lstm (LSTM)                 (None, 10)                4440
#  dense (Dense)               (None, 10)                110
#  dense_1 (Dense)             (None, 1)                 11
# =================================================================
# Total params: 7,661


# ----------------------임베딩3-------------------------------------------
model.add(Embedding(input_dim=10, output_dim=100))
model.add(LSTM(10))    #(None, 10) 
model.add(Dense(10))
model.add(Dense(1, activation='sigmoid'))
model.summary()

#3. 컴파일 및 훈련
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x, labels, epochs=100)


#4. 평가 및 예측
loss = model.evaluate(x, labels) 
result = model.predict(y)

print(np.round(result),'\n', loss)