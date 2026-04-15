import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
text1 = '나는 지금 진짜 진짜 매우 매우 맛있는 김밥을 엄청 마구 마구 마구 마구 먹었다.'
text2 = '태운이는 선생을 괴롭힌다. 준영이는 못생겼다. 사영이는 마구 마구 더 못생겼다'

# 맹그러봐

token = Tokenizer()
#인스턴스를 생성했다
token.fit_on_texts([text1])
token.fit_on_texts([text2])

x1 = token.texts_to_sequences([text1])
x2 = token.texts_to_sequences([text2])

x1 = pd.DataFrame(x1)
x2 = pd.DataFrame(x2)
print(x1,x2)
print(x1.shape, x2.shape)
x = pd.concat([x1,x2], axis=1)
print(x.shape)
'''
x = pd.get_dummies(pd.Series(np.array(x).reshape(-1,)))
print(x)


'''
x = to_categorical(x)
print(x.shape)

# [[3, 4, 5, 2, 2, 6, 7, 8, 1, 1, 1, 1, 9]]
x = x[:, :, 1:]
x = x.reshape(24,16)
print(x)
################# 원핫 3가지 맹글어봐!!############
#갯더미 원핫 인코더 

'''
ohe = OneHotEncoder(sparse=False)
x = np.array(x).reshape(-1,1) 
ohe.fit(x)
x_encoded3 = ohe.transform(x)   
print(x_encoded3)



###################
x_encoded2 = ohe.fit_transform(x)
print(x_encoded2)
'''