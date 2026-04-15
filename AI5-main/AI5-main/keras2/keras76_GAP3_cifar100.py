<<<<<<< HEAD
<<<<<<< HEAD
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

tf.random.set_seed(333)
np.random.seed(333)
print(tf.__version__)   # 2.7.4

from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
x_train = x_train.reshape(50000,3072,1)
x_test = x_test.reshape(10000,3072,1)
print(np.unique(y_train, return_counts=True))

x_train = x_train/255.
x_test = x_test/255.

from sklearn.preprocessing import OneHotEncoder


# 원-핫 인코딩 변환
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

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

'''
#Flatten_____________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 vgg16 (Functional)          (None, 7, 7, 512)         14714688

 flatten (Flatten)           (None, 25088)             0

 dense (Dense)               (None, 100)               2508900

 dense_1 (Dense)             (None, 100)               10100

 dense_2 (Dense)             (None, 10)                1010

=================================================================
Total params: 17,234,698
Trainable params: 2,520,010
Non-trainable params: 14,714,688
_________________________________________________________________

0  <keras.engine.functional.Functional object at 0x000002ABA92D0220>  vgg16      False
1  <keras.layers.core.flatten.Flatten object at 0x000002ABA92951C0>   flatten    True
2  <keras.layers.core.dense.Dense object at 0x000002ABA9317940>       dense      True
3  <keras.layers.core.dense.Dense object at 0x000002ABA9317F40>       dense_1    True
4  <keras.layers.core.dense.Dense object at 0x000002ABA9326940>       dense_2    True

#GAP
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 vgg16 (Functional)          (None, 7, 7, 512)         14714688

 global_average_pooling2d (G  (None, 512)              0
 lobalAveragePooling2D)

 dense (Dense)               (None, 100)               51300

 dense_1 (Dense)             (None, 100)               10100

 dense_2 (Dense)             (None, 10)                1010

=================================================================
Total params: 14,777,098
Trainable params: 62,410
Non-trainable params: 14,714,688
_________________________________________________________________
0  <keras.engine.functional.Functional object at 0x000001D2BA0E01F0>           vgg16                     False
1  <keras.layers.pooling.GlobalAveragePooling2D object at 0x000001D2B99D5190>  global_average_pooling2d  True
2  <keras.layers.core.dense.Dense object at 0x000001D2BA127910>                dense                     True
3  <keras.layers.core.dense.Dense object at 0x000001D2BA127E80>                dense_1                   True
4  <keras.layers.core.dense.Dense object at 0x000001D2BA137910>                dense_2                   True
'''

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
from tensorflow.keras.utils import to_categorical

tf.random.set_seed(333)
np.random.seed(333)
print(tf.__version__)   # 2.7.4

from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
x_train = x_train.reshape(50000,3072,1)
x_test = x_test.reshape(10000,3072,1)
print(np.unique(y_train, return_counts=True))

x_train = x_train/255.
x_test = x_test/255.

from sklearn.preprocessing import OneHotEncoder


# 원-핫 인코딩 변환
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

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

'''
#Flatten_____________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 vgg16 (Functional)          (None, 7, 7, 512)         14714688

 flatten (Flatten)           (None, 25088)             0

 dense (Dense)               (None, 100)               2508900

 dense_1 (Dense)             (None, 100)               10100

 dense_2 (Dense)             (None, 10)                1010

=================================================================
Total params: 17,234,698
Trainable params: 2,520,010
Non-trainable params: 14,714,688
_________________________________________________________________

0  <keras.engine.functional.Functional object at 0x000002ABA92D0220>  vgg16      False
1  <keras.layers.core.flatten.Flatten object at 0x000002ABA92951C0>   flatten    True
2  <keras.layers.core.dense.Dense object at 0x000002ABA9317940>       dense      True
3  <keras.layers.core.dense.Dense object at 0x000002ABA9317F40>       dense_1    True
4  <keras.layers.core.dense.Dense object at 0x000002ABA9326940>       dense_2    True

#GAP
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 vgg16 (Functional)          (None, 7, 7, 512)         14714688

 global_average_pooling2d (G  (None, 512)              0
 lobalAveragePooling2D)

 dense (Dense)               (None, 100)               51300

 dense_1 (Dense)             (None, 100)               10100

 dense_2 (Dense)             (None, 10)                1010

=================================================================
Total params: 14,777,098
Trainable params: 62,410
Non-trainable params: 14,714,688
_________________________________________________________________
0  <keras.engine.functional.Functional object at 0x000001D2BA0E01F0>           vgg16                     False
1  <keras.layers.pooling.GlobalAveragePooling2D object at 0x000001D2B99D5190>  global_average_pooling2d  True
2  <keras.layers.core.dense.Dense object at 0x000001D2BA127910>                dense                     True
3  <keras.layers.core.dense.Dense object at 0x000001D2BA127E80>                dense_1                   True
4  <keras.layers.core.dense.Dense object at 0x000001D2BA137910>                dense_2                   True
'''

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
from tensorflow.keras.utils import to_categorical

tf.random.set_seed(333)
np.random.seed(333)
print(tf.__version__)   # 2.7.4

from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
x_train = x_train.reshape(50000,3072,1)
x_test = x_test.reshape(10000,3072,1)
print(np.unique(y_train, return_counts=True))

x_train = x_train/255.
x_test = x_test/255.

from sklearn.preprocessing import OneHotEncoder


# 원-핫 인코딩 변환
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

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

'''
#Flatten_____________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 vgg16 (Functional)          (None, 7, 7, 512)         14714688

 flatten (Flatten)           (None, 25088)             0

 dense (Dense)               (None, 100)               2508900

 dense_1 (Dense)             (None, 100)               10100

 dense_2 (Dense)             (None, 10)                1010

=================================================================
Total params: 17,234,698
Trainable params: 2,520,010
Non-trainable params: 14,714,688
_________________________________________________________________

0  <keras.engine.functional.Functional object at 0x000002ABA92D0220>  vgg16      False
1  <keras.layers.core.flatten.Flatten object at 0x000002ABA92951C0>   flatten    True
2  <keras.layers.core.dense.Dense object at 0x000002ABA9317940>       dense      True
3  <keras.layers.core.dense.Dense object at 0x000002ABA9317F40>       dense_1    True
4  <keras.layers.core.dense.Dense object at 0x000002ABA9326940>       dense_2    True

#GAP
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 vgg16 (Functional)          (None, 7, 7, 512)         14714688

 global_average_pooling2d (G  (None, 512)              0
 lobalAveragePooling2D)

 dense (Dense)               (None, 100)               51300

 dense_1 (Dense)             (None, 100)               10100

 dense_2 (Dense)             (None, 10)                1010

=================================================================
Total params: 14,777,098
Trainable params: 62,410
Non-trainable params: 14,714,688
_________________________________________________________________
0  <keras.engine.functional.Functional object at 0x000001D2BA0E01F0>           vgg16                     False
1  <keras.layers.pooling.GlobalAveragePooling2D object at 0x000001D2B99D5190>  global_average_pooling2d  True
2  <keras.layers.core.dense.Dense object at 0x000001D2BA127910>                dense                     True
3  <keras.layers.core.dense.Dense object at 0x000001D2BA127E80>                dense_1                   True
4  <keras.layers.core.dense.Dense object at 0x000001D2BA137910>                dense_2                   True
'''

import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
print(results)

>>>>>>> 70eabacb3fa0ad4089229f1c83ce2c346b0e48a8
