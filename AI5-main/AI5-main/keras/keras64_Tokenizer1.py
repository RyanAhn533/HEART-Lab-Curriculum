#프롬포트 엔지니어링 vs 하이퍼 파라미터 튜닝의 차이점
#모델링을 할 때 모델링을 바꿔주는거 / 프롬포트 엔지니어링은 챗지피티같은거 효율적으로 쓰게 해주는 거
#양방향 트렌스포머 Bidirection  Renual regressor? X or 문제 -> 인공지능의 겨울이 왔다
#legacy한 자연어 처리에 대해서 조금 해보겠다

#이미지를 조각조각 token화 시켜서 데이터로 사용하겠다.
#빛코딩

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
text = '나는 지금 진짜 진짜 매우 매우 맛있는 김밥을 엄청 마구 마구 마구 마구 먹었다.'
#text = '성우하이텍 떡상 가즈아 영차 영차 영차 영차 영차 영차 영차 영차 영차 '
#이것을 어떻게 수치화 시킬 것 인가? - 띄어쓰기 단위로

token = Tokenizer()
#인스턴스를 생성했다
token.fit_on_texts([text])

print(token.word_index)
#{'마구': 1, '매우': 2, '나는': 3, '지금': 4, '진짜': 5, '맛있는': 6, '김밥을': 7, '엄청': 8, '먹었다': 9}
# {'영차': 1, '성우하이텍': 2, '떡상': 3, '가즈아': 4}
#첫번째는 많이 나오는 순서, 뒤에는 먼저 나오는 순서



#1 데이터 분석하고 trian data 하고 
# 랜덤 포레스트 모델 학습 = DNN 모델 썼다
# Regressor=회귀



# print(token.word_counts)
# OrderedDict([('성우하이텍', 1), ('떡상', 1), ('가즈아', 1), ('영차', 9)])

x = token.texts_to_sequences([text])


x = pd.DataFrame(x)
print(x)
x = x.to_numpy()
print(x)
print(x.shape)
x = x.reshape(14,1)
print(x)
'''
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
x = to_categorical(x)
print(x.shape)

# [[3, 4, 5, 2, 2, 6, 7, 8, 1, 1, 1, 1, 9]]
x = x[:, :, 1:]
print(x.shape)
x = x.reshape(14,9)

################# 원핫 3가지 맹글어봐!!############
#갯더미 원핫 인코더 



ohe = OneHotEncoder(sparse=False)
x = np.array(x).reshape(-1,1) 
ohe.fit(x)
x_encoded3 = ohe.transform(x)   
print(x_encoded3)


###################
x_encoded2 = ohe.fit_transform(x)
print(x_encoded2)

'''
###################
#x = pd.get_dummies(sum(x, []))
x = pd.get_dummies(pd.Series(np.array(x).reshape(-1,)))
print(x)

