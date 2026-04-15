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
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

#2. 모델 구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

model.summary()

print(model.weights)
'''
 Layer (type)                Output Shape              Param #
=================================================================
 dense (Dense)               (None, 3)                 6

 dense_1 (Dense)             (None, 2)                 8

 dense_2 (Dense)             (None, 1)                 3

=================================================================
Total params: 17
Trainable params: 17
Non-trainable params: 0
_________________________________________________________________
[<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[ 0.13603318, -0.03480017,  0.7743634 ]], dtype=float32)>, <tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, <tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, numpy=
array([[-0.92561173,  0.8256177 ],
       [ 0.6200088 ,  1.0182774 ],
       [-0.5191052 , -0.6304303 ]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>, <tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy=        

array([[-0.02628279],
       [-1.074922  ]], dtype=float32)>, <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]

'''

print('==============================')
print(model.trainable_weights)          # 동일
print('==============================')
print(len(model.weights))               # 6
print(len(model.trainable_weights))     # 6

######################################################################
model.trainable = False                 # 동결 ★ ★ ★ ★ ★
######################################################################
print(len(model.weights))               # 6
print(len(model.trainable_weights))     # 0

print('=============== model.weights ===============')
print(model.weights)                    # 동일
print('=============== trainable_weights ===============')
print(model.trainable_weights)          # []

model.summary()
'''
 Layer (type)                Output Shape              Param #
=================================================================
 dense (Dense)               (None, 3)                 6

 dense_1 (Dense)             (None, 2)                 8

 dense_2 (Dense)             (None, 1)                 3

=================================================================
Total params: 17
Trainable params: 0             *
Non-trainable params: 17        *
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
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

#2. 모델 구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

model.summary()

print(model.weights)
'''
 Layer (type)                Output Shape              Param #
=================================================================
 dense (Dense)               (None, 3)                 6

 dense_1 (Dense)             (None, 2)                 8

 dense_2 (Dense)             (None, 1)                 3

=================================================================
Total params: 17
Trainable params: 17
Non-trainable params: 0
_________________________________________________________________
[<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[ 0.13603318, -0.03480017,  0.7743634 ]], dtype=float32)>, <tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, <tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, numpy=
array([[-0.92561173,  0.8256177 ],
       [ 0.6200088 ,  1.0182774 ],
       [-0.5191052 , -0.6304303 ]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>, <tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy=        

array([[-0.02628279],
       [-1.074922  ]], dtype=float32)>, <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]

'''

print('==============================')
print(model.trainable_weights)          # 동일
print('==============================')
print(len(model.weights))               # 6
print(len(model.trainable_weights))     # 6

######################################################################
model.trainable = False                 # 동결 ★ ★ ★ ★ ★
######################################################################
print(len(model.weights))               # 6
print(len(model.trainable_weights))     # 0

print('=============== model.weights ===============')
print(model.weights)                    # 동일
print('=============== trainable_weights ===============')
print(model.trainable_weights)          # []

model.summary()
'''
 Layer (type)                Output Shape              Param #
=================================================================
 dense (Dense)               (None, 3)                 6

 dense_1 (Dense)             (None, 2)                 8

 dense_2 (Dense)             (None, 1)                 3

=================================================================
Total params: 17
Trainable params: 0             *
Non-trainable params: 17        *
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
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

#2. 모델 구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

model.summary()

print(model.weights)
'''
 Layer (type)                Output Shape              Param #
=================================================================
 dense (Dense)               (None, 3)                 6

 dense_1 (Dense)             (None, 2)                 8

 dense_2 (Dense)             (None, 1)                 3

=================================================================
Total params: 17
Trainable params: 17
Non-trainable params: 0
_________________________________________________________________
[<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[ 0.13603318, -0.03480017,  0.7743634 ]], dtype=float32)>, <tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, <tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, numpy=
array([[-0.92561173,  0.8256177 ],
       [ 0.6200088 ,  1.0182774 ],
       [-0.5191052 , -0.6304303 ]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>, <tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy=        

array([[-0.02628279],
       [-1.074922  ]], dtype=float32)>, <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]

'''

print('==============================')
print(model.trainable_weights)          # 동일
print('==============================')
print(len(model.weights))               # 6
print(len(model.trainable_weights))     # 6

######################################################################
model.trainable = False                 # 동결 ★ ★ ★ ★ ★
######################################################################
print(len(model.weights))               # 6
print(len(model.trainable_weights))     # 0

print('=============== model.weights ===============')
print(model.weights)                    # 동일
print('=============== trainable_weights ===============')
print(model.trainable_weights)          # []

model.summary()
'''
 Layer (type)                Output Shape              Param #
=================================================================
 dense (Dense)               (None, 3)                 6

 dense_1 (Dense)             (None, 2)                 8

 dense_2 (Dense)             (None, 1)                 3

=================================================================
Total params: 17
Trainable params: 0             *
Non-trainable params: 17        *
'''
>>>>>>> 70eabacb3fa0ad4089229f1c83ce2c346b0e48a8
