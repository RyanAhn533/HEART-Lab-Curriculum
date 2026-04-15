<<<<<<< HEAD
<<<<<<< HEAD
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import tensorflow as tf

tf.random.set_seed(333)
np.random.seed(333)
print(tf.__version__)   # 2.7.4

from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data() 
x_train = x_train.reshape(50000,1024,3)
x_test = x_test.reshape(10000,1024,3)
x_train = x_train/255.
x_test = x_test/255.

vgg16 = VGG16(# weights='imagenet',
              include_top=False,
              input_shape=(32, 32 ,3),
              )

vgg16.trainable = False     # 가중치 동결

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

model.summary()

'''
 Layer (type)                Output Shape              Param #
=================================================================
 vgg16 (Functional)          (None, 1, 1, 512)         14714688
 flatten (Flatten)           (None, 512)               0
 dense (Dense)               (None, 100)               51300
 dense_1 (Dense)             (None, 100)               10100
 dense_2 (Dense)             (None, 10)                1010
=================================================================
Total params: 14,777,098
Trainable params: 14,777,098
Non-trainable params: 0
'''

# vgg16.trainable = False 추가
'''
 Layer (type)                Output Shape              Param #
=================================================================
 vgg16 (Functional)          (None, 1, 1, 512)         14714688
 flatten (Flatten)           (None, 512)               0
 dense (Dense)               (None, 100)               51300
 dense_1 (Dense)             (None, 100)               10100
 dense_2 (Dense)             (None, 10)                1010
=================================================================
Total params: 14,777,098
Trainable params: 62,410
Non-trainable params: 14,714,688
'''

########## [실습] 3가지 비교하기 ##########
# 1. 이전에 본인이 한 최상의 결과
# 2. 가중치를 동결하지 않고 훈련시켰을 때, trainable=True 
# 3. 가중치를 동결하고 훈련시켰을 때, trainable=False
# 위의 2, 3번은 time 체크 까지
=======
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import tensorflow as tf

tf.random.set_seed(333)
np.random.seed(333)
print(tf.__version__)   # 2.7.4

from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data() 
x_train = x_train.reshape(50000,1024,3)
x_test = x_test.reshape(10000,1024,3)
x_train = x_train/255.
x_test = x_test/255.

vgg16 = VGG16(# weights='imagenet',
              include_top=False,
              input_shape=(32, 32 ,3),
              )

vgg16.trainable = False     # 가중치 동결

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

model.summary()

'''
 Layer (type)                Output Shape              Param #
=================================================================
 vgg16 (Functional)          (None, 1, 1, 512)         14714688
 flatten (Flatten)           (None, 512)               0
 dense (Dense)               (None, 100)               51300
 dense_1 (Dense)             (None, 100)               10100
 dense_2 (Dense)             (None, 10)                1010
=================================================================
Total params: 14,777,098
Trainable params: 14,777,098
Non-trainable params: 0
'''

# vgg16.trainable = False 추가
'''
 Layer (type)                Output Shape              Param #
=================================================================
 vgg16 (Functional)          (None, 1, 1, 512)         14714688
 flatten (Flatten)           (None, 512)               0
 dense (Dense)               (None, 100)               51300
 dense_1 (Dense)             (None, 100)               10100
 dense_2 (Dense)             (None, 10)                1010
=================================================================
Total params: 14,777,098
Trainable params: 62,410
Non-trainable params: 14,714,688
'''

########## [실습] 3가지 비교하기 ##########
# 1. 이전에 본인이 한 최상의 결과
# 2. 가중치를 동결하지 않고 훈련시켰을 때, trainable=True 
# 3. 가중치를 동결하고 훈련시켰을 때, trainable=False
# 위의 2, 3번은 time 체크 까지
>>>>>>> cd855f8 (message)
=======
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import tensorflow as tf

tf.random.set_seed(333)
np.random.seed(333)
print(tf.__version__)   # 2.7.4

from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data() 
x_train = x_train.reshape(50000,1024,3)
x_test = x_test.reshape(10000,1024,3)
x_train = x_train/255.
x_test = x_test/255.

vgg16 = VGG16(# weights='imagenet',
              include_top=False,
              input_shape=(32, 32 ,3),
              )

vgg16.trainable = False     # 가중치 동결

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

model.summary()

'''
 Layer (type)                Output Shape              Param #
=================================================================
 vgg16 (Functional)          (None, 1, 1, 512)         14714688
 flatten (Flatten)           (None, 512)               0
 dense (Dense)               (None, 100)               51300
 dense_1 (Dense)             (None, 100)               10100
 dense_2 (Dense)             (None, 10)                1010
=================================================================
Total params: 14,777,098
Trainable params: 14,777,098
Non-trainable params: 0
'''

# vgg16.trainable = False 추가
'''
 Layer (type)                Output Shape              Param #
=================================================================
 vgg16 (Functional)          (None, 1, 1, 512)         14714688
 flatten (Flatten)           (None, 512)               0
 dense (Dense)               (None, 100)               51300
 dense_1 (Dense)             (None, 100)               10100
 dense_2 (Dense)             (None, 10)                1010
=================================================================
Total params: 14,777,098
Trainable params: 62,410
Non-trainable params: 14,714,688
'''

########## [실습] 3가지 비교하기 ##########
# 1. 이전에 본인이 한 최상의 결과
# 2. 가중치를 동결하지 않고 훈련시켰을 때, trainable=True 
# 3. 가중치를 동결하고 훈련시켰을 때, trainable=False
# 위의 2, 3번은 time 체크 까지
>>>>>>> 70eabacb3fa0ad4089229f1c83ce2c346b0e48a8
