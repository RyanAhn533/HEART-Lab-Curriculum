import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])


# [실습] keras04 의 가장 좋은 레이어와 노드를 이용하여,
# 최소의 loss를 맹그러
# batch_size 조절
# 에포는 100으로 고성을 풀어주겠노라!!! 
# 로스 기준 0.32 미만!!!!

#2. 모델구성
model = Sequential()
model.add(Dense(4, input_dim=1))
model.add(Dense(7, input_dim=8))
model.add(Dense(3, input_dim=7))
model.add(Dense(1, input_dim=3))

# 'epochs :  100
로스 :  0.33883342146873474
# 6의 예측값 :  [[5.6892266]]


,,,,

#2. 모델구성
model = Sequential()
model.add(Dense(4, input_dim=1))
model.add(Dense(7, input_dim=8))
model.add(Dense(3, input_dim=7))
model.add(Dense(1, input_dim=3))

epochs = 100
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=epochs, batch_size=8)




""""
- 공책 참조
[index (행에 대한 이름 값)]
이름  국어  영어  수학  과학
 홍    100   10    50     30            행, 열
 김                                       # (4, 3)  (4, )
 박                                            x      y는 4개의 데이터만 있음     
김2      !      !      !      ?
 ㅇ                           ?           # 프리행열
 ㅇ                           ?             (3, 3) (3, )
 
* 열, 컬럼, 피쳐, 특성
* 열이 중요함
행 하나가 사라져도 계산에 문제가 안되지만 열 (수학)이 사라지면
국어, 영어로만 계산하여 과학을 예측해야하기에 계산이 어려워짐

1 <- 스칼라
[1,2,3] <- 벡터  1차원  -> (3, )
[[1,2,3],[4,5,6]] <- 행렬   -> (2, 3)
[[[1,2,3],[4,5,6]]] <- tensorflow 다차원 행렬, 행렬보다 한단계 높음   -> (1,2,3)



print("x1 :", x1.shape)   # 어떤 모양
터미널 
x1 : (3,)


1. [1,2,3]   (3, )
2. [[1,2,3]]   (1,3)
3. [[1,2],[3,4]]   (2,2)
4. [[1,2],[3,4][5,6]]   (3,2)
5. [[[1,2],[3,4],[5,6]]   (3,2)
6. [[[1,2],[3,4]],[[5,6],[7,8]]]
7. [[[[1,2,3,4,5],[6,7,8,9,10]]]]   (1,1,2,8)
8. [[1,2,3],[4,5,6]]  (2,3)
9. [[[[1]]]],[[[[2]]]]   (2,1,1,1)""""