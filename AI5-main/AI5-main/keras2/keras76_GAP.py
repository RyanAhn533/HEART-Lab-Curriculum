<<<<<<< HEAD
<<<<<<< HEAD
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
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
              include_top=False, input_shape=(224,224,3)
              )

vgg16.trainable = False     # 가중치 동결

model = Sequential()
model.add(vgg16)
model.add(Flatten())
#model.add(GlobalAveragePooling2D()) #2바이1짜리는 Flatten이랑 비슷하다 !
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

model.summary()

"""
=================================================================
Total params: 14,777,098
Trainable params: 62,410
Non-trainable params: 14,714,688
_________________________________________________________________


=================================================================
Total params: 17,234,698
Trainable params: 2,520,010
Non-trainable params: 14,714,688
_________________________________________________________________

"""

import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
print(results)

=======
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
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
              include_top=False, input_shape=(224,224,3)
              )

vgg16.trainable = False     # 가중치 동결

model = Sequential()
model.add(vgg16)
model.add(Flatten())
#model.add(GlobalAveragePooling2D()) #2바이1짜리는 Flatten이랑 비슷하다 !
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

model.summary()

"""
=================================================================
Total params: 14,777,098
Trainable params: 62,410
Non-trainable params: 14,714,688
_________________________________________________________________


=================================================================
Total params: 17,234,698
Trainable params: 2,520,010
Non-trainable params: 14,714,688
_________________________________________________________________

"""

import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
print(results)

>>>>>>> cd855f8 (message)
=======
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
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
              include_top=False, input_shape=(224,224,3)
              )

vgg16.trainable = False     # 가중치 동결

model = Sequential()
model.add(vgg16)
model.add(Flatten())
#model.add(GlobalAveragePooling2D()) #2바이1짜리는 Flatten이랑 비슷하다 !
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

model.summary()

"""
=================================================================
Total params: 14,777,098
Trainable params: 62,410
Non-trainable params: 14,714,688
_________________________________________________________________


=================================================================
Total params: 17,234,698
Trainable params: 2,520,010
Non-trainable params: 14,714,688
_________________________________________________________________

"""

import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
print(results)

>>>>>>> 70eabacb3fa0ad4089229f1c83ce2c346b0e48a8
