import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input 

#1. 데이터           
 #<- 잘못된 형태        
x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1,1.1,1.2,1.3,1.4,1.5,1.6,1.5,1.4,1.3],
              [9,8,7,6,5,4,3,2,1,0]])  #벡터 2개짜리는 행렬 ->
             #이렇게되면 행열이 반대로 찍힘 
#x = np.array([[1,6],[2,7],[3,8],[4,9],[5,10]])
y = np.array([1,2,3,4,5,6,7,8,9,10])
x = x.T

print(x.shape)
print(y.shape)
"""
#2. 모델구성(순차형)
model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(9))
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(1))

model.summary()
#Sequential모델과 함수형 모델이 있다.
#Sequential 모델을 함수형으로 바꿀 수 있다.

Total params: 290
Trainable params: 290
Non-trainable params: 0
________________________

"""
#2-2 모델구성(함수형)
input1 = Input(shape=(3,))
dense1 = Dense(10)(input1, name='ys1')(input1)
dense2 = Dense(9)(dense1, name='ys2')(dense1)
dense3 = Dense(8)(dense2, name='ys3')(dense2)
dense4 = Dense(7)(dense3, name='ys4')(dense3)
output1 = Dense(1)(dense4, name='ys5')(dense4)
#시작 input1 ~ output1 여기까지가 모델이야라고 정해주는게 필요함 아래 모델
model = Model(inputs=input1, outputs=output1)

model.summary()

"""
Total params: 290
Trainable params: 290
Non-trainable params: 0
______________________________
"""

