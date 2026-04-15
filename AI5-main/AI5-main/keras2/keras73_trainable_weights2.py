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
model.trainable = False                 # 동결 ★ ★ ★ ★ ★
# model.trainable = True                  # 안동결 ★ ★ ★ ★ ★, Default
######################################################################

print('==============================')
print(model.weights)
print('==============================')

'''
[<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[ 0.13603318, -0.03480017,  0.7743634 ]], dtype=float32)>, <tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, <tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, numpy=
array([[-0.92561173,  0.8256177 ],
       [ 0.6200088 ,  1.0182774 ],
       [-0.5191052 , -0.6304303 ]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>, <tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy=       
array([[-0.02628279],
       [-1.074922  ]], dtype=float32)>, <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]
'''

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, batch_size=1, epochs=1000, verbose=0)

#4. 평가, 예측
y_predict = model.predict(x)
print(y_predict)
'''
주석 (model.trainable = True)
[[1.0000002]
 [2.       ]
 [2.9999995]
 [4.       ]
 [5.       ]]

model.trainable = False
[[0.45656443]
 [0.91312885]
 [1.369693  ]
 [1.8262577 ]
 [2.2828221 ]]
'''

# [실습] model.trainable = False 계산
'''
 [<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[ 0.13603318, -0.03480017,  0.7743634 ]], dtype=float32)>, <tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, <tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, numpy=
array([[-0.92561173,  0.8256177 ],
       [ 0.6200088 ,  1.0182774 ],
       [-0.5191052 , -0.6304303 ]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>, <tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy=       
array([[-0.02628279],
       [-1.074922  ]], dtype=float32)>, <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]
==============================
[[0.45656443]]
==============================
[ 0.13603318, -0.03480017,  0.7743634 ]
과
[[-0.92561173,  0.8256177 ],
[ 0.6200088 ,  1.0182774 ],
[-0.5191052 , -0.6304303 ]]
--------------------------------------------------
-0.1259139070772014
-0.021576411641496
-0.40197606762968

0.112311401195286
-0.035436226627158
-0.48818215057102
--------------------------------------------------
[-0.5494663863483774, -0.411306976002892]
과
[[-0.02628279],
[-1.074922  ]]
--------------------------------------------------
0.014441509644453270044946
0.442122917258980674424
--------------------------------------------------
0.456564426903433944468946
0.45656443
'''
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
model.trainable = False                 # 동결 ★ ★ ★ ★ ★
# model.trainable = True                  # 안동결 ★ ★ ★ ★ ★, Default
######################################################################

print('==============================')
print(model.weights)
print('==============================')

'''
[<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[ 0.13603318, -0.03480017,  0.7743634 ]], dtype=float32)>, <tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, <tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, numpy=
array([[-0.92561173,  0.8256177 ],
       [ 0.6200088 ,  1.0182774 ],
       [-0.5191052 , -0.6304303 ]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>, <tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy=       
array([[-0.02628279],
       [-1.074922  ]], dtype=float32)>, <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]
'''

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, batch_size=1, epochs=1000, verbose=0)

#4. 평가, 예측
y_predict = model.predict(x)
print(y_predict)
'''
주석 (model.trainable = True)
[[1.0000002]
 [2.       ]
 [2.9999995]
 [4.       ]
 [5.       ]]

model.trainable = False
[[0.45656443]
 [0.91312885]
 [1.369693  ]
 [1.8262577 ]
 [2.2828221 ]]
'''

# [실습] model.trainable = False 계산
'''
 [<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[ 0.13603318, -0.03480017,  0.7743634 ]], dtype=float32)>, <tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, <tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, numpy=
array([[-0.92561173,  0.8256177 ],
       [ 0.6200088 ,  1.0182774 ],
       [-0.5191052 , -0.6304303 ]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>, <tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy=       
array([[-0.02628279],
       [-1.074922  ]], dtype=float32)>, <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]
==============================
[[0.45656443]]
==============================
[ 0.13603318, -0.03480017,  0.7743634 ]
과
[[-0.92561173,  0.8256177 ],
[ 0.6200088 ,  1.0182774 ],
[-0.5191052 , -0.6304303 ]]
--------------------------------------------------
-0.1259139070772014
-0.021576411641496
-0.40197606762968

0.112311401195286
-0.035436226627158
-0.48818215057102
--------------------------------------------------
[-0.5494663863483774, -0.411306976002892]
과
[[-0.02628279],
[-1.074922  ]]
--------------------------------------------------
0.014441509644453270044946
0.442122917258980674424
--------------------------------------------------
0.456564426903433944468946
0.45656443
'''
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
model.trainable = False                 # 동결 ★ ★ ★ ★ ★
# model.trainable = True                  # 안동결 ★ ★ ★ ★ ★, Default
######################################################################

print('==============================')
print(model.weights)
print('==============================')

'''
[<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[ 0.13603318, -0.03480017,  0.7743634 ]], dtype=float32)>, <tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, <tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, numpy=
array([[-0.92561173,  0.8256177 ],
       [ 0.6200088 ,  1.0182774 ],
       [-0.5191052 , -0.6304303 ]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>, <tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy=       
array([[-0.02628279],
       [-1.074922  ]], dtype=float32)>, <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]
'''

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, batch_size=1, epochs=1000, verbose=0)

#4. 평가, 예측
y_predict = model.predict(x)
print(y_predict)
'''
주석 (model.trainable = True)
[[1.0000002]
 [2.       ]
 [2.9999995]
 [4.       ]
 [5.       ]]

model.trainable = False
[[0.45656443]
 [0.91312885]
 [1.369693  ]
 [1.8262577 ]
 [2.2828221 ]]
'''

# [실습] model.trainable = False 계산
'''
 [<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[ 0.13603318, -0.03480017,  0.7743634 ]], dtype=float32)>, <tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, <tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, numpy=
array([[-0.92561173,  0.8256177 ],
       [ 0.6200088 ,  1.0182774 ],
       [-0.5191052 , -0.6304303 ]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>, <tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy=       
array([[-0.02628279],
       [-1.074922  ]], dtype=float32)>, <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]
==============================
[[0.45656443]]
==============================
[ 0.13603318, -0.03480017,  0.7743634 ]
과
[[-0.92561173,  0.8256177 ],
[ 0.6200088 ,  1.0182774 ],
[-0.5191052 , -0.6304303 ]]
--------------------------------------------------
-0.1259139070772014
-0.021576411641496
-0.40197606762968

0.112311401195286
-0.035436226627158
-0.48818215057102
--------------------------------------------------
[-0.5494663863483774, -0.411306976002892]
과
[[-0.02628279],
[-1.074922  ]]
--------------------------------------------------
0.014441509644453270044946
0.442122917258980674424
--------------------------------------------------
0.456564426903433944468946
0.45656443
'''
>>>>>>> 70eabacb3fa0ad4089229f1c83ce2c346b0e48a8
