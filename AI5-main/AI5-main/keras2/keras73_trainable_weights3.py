<<<<<<< HEAD
<<<<<<< HEAD
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

tf.random.set_seed(333)
np.random.seed(333)
print(tf.__version__)       # 2.7.4

#1. 데이터
# x = np.array([1,2,3,4,5])
# y = np.array([1,2,3,4,5])
x = np.array([1])
y = np.array([1])

#2. 모델 구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

######################################################################
# model.trainable = False                 # 동결 ★ ★ ★ ★ ★
model.trainable = True                # 안동결 ★ ★ ★ ★ ★, Default
######################################################################

print('==============================')
print(model.weights)
print('==============================')

'''
[[1.]]

[<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[ 0.08451437, -0.08841538,  0.8282983 ]], dtype=float32)>, 
<tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([-0.05151882, -0.05361514,  0.05393542], dtype=float32)>, 

<tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, numpy=
array([[-0.9780189 ,  0.7826858 ],
       [ 0.69753873,  1.0858245 ],
       [-0.5847604 , -0.68446755]], dtype=float32)>, 
       <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([-0.06426805, -0.05272118], dtype=float32)>, 
       
<tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy=
array([[-0.08096294],
       [-1.1340268 ]], dtype=float32)>, 
       <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.05221008], dtype=float32)>]
'''




#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, batch_size=1, epochs=1000, verbose=0)

#4. 평가, 예측
y_predict = model.predict(x)
print(y_predict)



=======
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

tf.random.set_seed(333)
np.random.seed(333)
print(tf.__version__)       # 2.7.4

#1. 데이터
# x = np.array([1,2,3,4,5])
# y = np.array([1,2,3,4,5])
x = np.array([1])
y = np.array([1])

#2. 모델 구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

######################################################################
# model.trainable = False                 # 동결 ★ ★ ★ ★ ★
model.trainable = True                # 안동결 ★ ★ ★ ★ ★, Default
######################################################################

print('==============================')
print(model.weights)
print('==============================')

'''
[[1.]]

[<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[ 0.08451437, -0.08841538,  0.8282983 ]], dtype=float32)>, 
<tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([-0.05151882, -0.05361514,  0.05393542], dtype=float32)>, 

<tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, numpy=
array([[-0.9780189 ,  0.7826858 ],
       [ 0.69753873,  1.0858245 ],
       [-0.5847604 , -0.68446755]], dtype=float32)>, 
       <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([-0.06426805, -0.05272118], dtype=float32)>, 
       
<tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy=
array([[-0.08096294],
       [-1.1340268 ]], dtype=float32)>, 
       <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.05221008], dtype=float32)>]
'''




#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, batch_size=1, epochs=1000, verbose=0)

#4. 평가, 예측
y_predict = model.predict(x)
print(y_predict)



>>>>>>> cd855f8 (message)
=======
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

tf.random.set_seed(333)
np.random.seed(333)
print(tf.__version__)       # 2.7.4

#1. 데이터
# x = np.array([1,2,3,4,5])
# y = np.array([1,2,3,4,5])
x = np.array([1])
y = np.array([1])

#2. 모델 구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

######################################################################
# model.trainable = False                 # 동결 ★ ★ ★ ★ ★
model.trainable = True                # 안동결 ★ ★ ★ ★ ★, Default
######################################################################

print('==============================')
print(model.weights)
print('==============================')

'''
[[1.]]

[<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[ 0.08451437, -0.08841538,  0.8282983 ]], dtype=float32)>, 
<tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([-0.05151882, -0.05361514,  0.05393542], dtype=float32)>, 

<tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, numpy=
array([[-0.9780189 ,  0.7826858 ],
       [ 0.69753873,  1.0858245 ],
       [-0.5847604 , -0.68446755]], dtype=float32)>, 
       <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([-0.06426805, -0.05272118], dtype=float32)>, 
       
<tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy=
array([[-0.08096294],
       [-1.1340268 ]], dtype=float32)>, 
       <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.05221008], dtype=float32)>]
'''




#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, batch_size=1, epochs=1000, verbose=0)

#4. 평가, 예측
y_predict = model.predict(x)
print(y_predict)



>>>>>>> 70eabacb3fa0ad4089229f1c83ce2c346b0e48a8
