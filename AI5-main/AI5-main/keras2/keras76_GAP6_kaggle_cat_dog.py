<<<<<<< HEAD
<<<<<<< HEAD
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization, MaxPool2D
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D

train_datagen = ImageDataGenerator(
    rescale=1/255,

horizontal_flip=True, #수평 뒤집기
    vertical_flip=True, #수직뒤집기
    width_shift_range=0.1, #평행이동
    height_shift_range=0.1, #평행이동 수직
    rotation_range=5, #정해진 각도만큼 이미지 회전
    zoom_range=1.2, #축소 또는 확대
    shear_range=0.7, # 좌표
    fill_mode="nearest", #비율에 맞춰서 채워라
    )


start = time.time()

np_path = 'C:\\프로그램\\ai5\\_data\\_save\\save_npy\\'

x_train = np.load(np_path + "keras43_01_x_train1.npy")
y_train = np.load(np_path + "keras43_01_y_train1.npy")
x_test = np.load(np_path + "keras43_01_x_test.npy")
y_test = np.load(np_path + "keras43_01_y_test.npy")
x_train = x_train.reshape(25000,100*100,3)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.9, random_state=1186)
print(x_train)
print(x_train.shape)

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(100, 100, 3))  # input_shape을 이미지 크기와 일치시킴

# 모델 생성
model = Sequential()
model.add(vgg16)          # VGG16을 모델에 추가
model.add(Flatten())      # Flatten을 추가하여 1차원으로 평탄화
#model.add(GlobalAveragePooling2D())
model.add(Dense(100, activation='relu'))  # 완전 연결층 추가
model.add(Dense(1, activation='sigmoid')) # 이진 분류를 위한 출력층

model.summary()  # 모델 구조 출력하여 각 레이어의 출력 형상 확인

'''
GAP
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 vgg16 (Functional)          (None, 3, 3, 512)         14714688

 global_average_pooling2d (G  (None, 512)              0
 lobalAveragePooling2D)

 dense (Dense)               (None, 100)               51300

 dense_1 (Dense)             (None, 1)                 101

=================================================================
Total params: 14,766,089
Trainable params: 14,766,089
Non-trainable params: 0
_________________________________________________________________
0  <keras.engine.functional.Functional object at 0x00000189466F9130>           vgg16                     True
1  <keras.layers.pooling.GlobalAveragePooling2D object at 0x00000189466F4250>  global_average_pooling2d  True
2  <keras.layers.core.dense.Dense object at 0x0000018946B51E50>                dense                     True
3  <keras.layers.core.dense.Dense object at 0x0000018946B58940>                dense_1                   True

Flatten
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 vgg16 (Functional)          (None, 3, 3, 512)         14714688

 flatten (Flatten)           (None, 4608)              0

 dense (Dense)               (None, 100)               460900

 dense_1 (Dense)             (None, 1)                 101

=================================================================
Total params: 15,175,689
Trainable params: 15,175,689
Non-trainable params: 0
_________________________________________________________________
c:\프로그램\ai5\study\keras2\keras76_GAP6_kaggle_cat_dog.py:85: FutureWarning: Passing a negative integer is deprecated in version 1.0 and will not be supported in future version. Instead, use None to not limit the column width.
  pd.set_option('max_colwidth', -1)
                                                          Layer Type Layer Name  Layer Trainable
0  <keras.engine.functional.Functional object at 0x000001FAAB4E9160>  vgg16      True
1  <keras.layers.core.flatten.Flatten object at 0x000001FAAB4E4280>   flatten    True
2  <keras.layers.core.dense.Dense object at 0x000001FAAB651FA0>       dense      True
3  <keras.layers.core.dense.Dense object at 0x000001FAAB658580>       dense_1    True
'''

import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
print(results)
=======
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization, MaxPool2D
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D

train_datagen = ImageDataGenerator(
    rescale=1/255,

horizontal_flip=True, #수평 뒤집기
    vertical_flip=True, #수직뒤집기
    width_shift_range=0.1, #평행이동
    height_shift_range=0.1, #평행이동 수직
    rotation_range=5, #정해진 각도만큼 이미지 회전
    zoom_range=1.2, #축소 또는 확대
    shear_range=0.7, # 좌표
    fill_mode="nearest", #비율에 맞춰서 채워라
    )


start = time.time()

np_path = 'C:\\프로그램\\ai5\\_data\\_save\\save_npy\\'

x_train = np.load(np_path + "keras43_01_x_train1.npy")
y_train = np.load(np_path + "keras43_01_y_train1.npy")
x_test = np.load(np_path + "keras43_01_x_test.npy")
y_test = np.load(np_path + "keras43_01_y_test.npy")
x_train = x_train.reshape(25000,100*100,3)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.9, random_state=1186)
print(x_train)
print(x_train.shape)

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(100, 100, 3))  # input_shape을 이미지 크기와 일치시킴

# 모델 생성
model = Sequential()
model.add(vgg16)          # VGG16을 모델에 추가
model.add(Flatten())      # Flatten을 추가하여 1차원으로 평탄화
#model.add(GlobalAveragePooling2D())
model.add(Dense(100, activation='relu'))  # 완전 연결층 추가
model.add(Dense(1, activation='sigmoid')) # 이진 분류를 위한 출력층

model.summary()  # 모델 구조 출력하여 각 레이어의 출력 형상 확인

'''
GAP
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 vgg16 (Functional)          (None, 3, 3, 512)         14714688

 global_average_pooling2d (G  (None, 512)              0
 lobalAveragePooling2D)

 dense (Dense)               (None, 100)               51300

 dense_1 (Dense)             (None, 1)                 101

=================================================================
Total params: 14,766,089
Trainable params: 14,766,089
Non-trainable params: 0
_________________________________________________________________
0  <keras.engine.functional.Functional object at 0x00000189466F9130>           vgg16                     True
1  <keras.layers.pooling.GlobalAveragePooling2D object at 0x00000189466F4250>  global_average_pooling2d  True
2  <keras.layers.core.dense.Dense object at 0x0000018946B51E50>                dense                     True
3  <keras.layers.core.dense.Dense object at 0x0000018946B58940>                dense_1                   True

Flatten
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 vgg16 (Functional)          (None, 3, 3, 512)         14714688

 flatten (Flatten)           (None, 4608)              0

 dense (Dense)               (None, 100)               460900

 dense_1 (Dense)             (None, 1)                 101

=================================================================
Total params: 15,175,689
Trainable params: 15,175,689
Non-trainable params: 0
_________________________________________________________________
c:\프로그램\ai5\study\keras2\keras76_GAP6_kaggle_cat_dog.py:85: FutureWarning: Passing a negative integer is deprecated in version 1.0 and will not be supported in future version. Instead, use None to not limit the column width.
  pd.set_option('max_colwidth', -1)
                                                          Layer Type Layer Name  Layer Trainable
0  <keras.engine.functional.Functional object at 0x000001FAAB4E9160>  vgg16      True
1  <keras.layers.core.flatten.Flatten object at 0x000001FAAB4E4280>   flatten    True
2  <keras.layers.core.dense.Dense object at 0x000001FAAB651FA0>       dense      True
3  <keras.layers.core.dense.Dense object at 0x000001FAAB658580>       dense_1    True
'''

import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
print(results)
>>>>>>> cd855f8 (message)
=======
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization, MaxPool2D
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D

train_datagen = ImageDataGenerator(
    rescale=1/255,

horizontal_flip=True, #수평 뒤집기
    vertical_flip=True, #수직뒤집기
    width_shift_range=0.1, #평행이동
    height_shift_range=0.1, #평행이동 수직
    rotation_range=5, #정해진 각도만큼 이미지 회전
    zoom_range=1.2, #축소 또는 확대
    shear_range=0.7, # 좌표
    fill_mode="nearest", #비율에 맞춰서 채워라
    )


start = time.time()

np_path = 'C:\\프로그램\\ai5\\_data\\_save\\save_npy\\'

x_train = np.load(np_path + "keras43_01_x_train1.npy")
y_train = np.load(np_path + "keras43_01_y_train1.npy")
x_test = np.load(np_path + "keras43_01_x_test.npy")
y_test = np.load(np_path + "keras43_01_y_test.npy")
x_train = x_train.reshape(25000,100*100,3)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.9, random_state=1186)
print(x_train)
print(x_train.shape)

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(100, 100, 3))  # input_shape을 이미지 크기와 일치시킴

# 모델 생성
model = Sequential()
model.add(vgg16)          # VGG16을 모델에 추가
model.add(Flatten())      # Flatten을 추가하여 1차원으로 평탄화
#model.add(GlobalAveragePooling2D())
model.add(Dense(100, activation='relu'))  # 완전 연결층 추가
model.add(Dense(1, activation='sigmoid')) # 이진 분류를 위한 출력층

model.summary()  # 모델 구조 출력하여 각 레이어의 출력 형상 확인

'''
GAP
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 vgg16 (Functional)          (None, 3, 3, 512)         14714688

 global_average_pooling2d (G  (None, 512)              0
 lobalAveragePooling2D)

 dense (Dense)               (None, 100)               51300

 dense_1 (Dense)             (None, 1)                 101

=================================================================
Total params: 14,766,089
Trainable params: 14,766,089
Non-trainable params: 0
_________________________________________________________________
0  <keras.engine.functional.Functional object at 0x00000189466F9130>           vgg16                     True
1  <keras.layers.pooling.GlobalAveragePooling2D object at 0x00000189466F4250>  global_average_pooling2d  True
2  <keras.layers.core.dense.Dense object at 0x0000018946B51E50>                dense                     True
3  <keras.layers.core.dense.Dense object at 0x0000018946B58940>                dense_1                   True

Flatten
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 vgg16 (Functional)          (None, 3, 3, 512)         14714688

 flatten (Flatten)           (None, 4608)              0

 dense (Dense)               (None, 100)               460900

 dense_1 (Dense)             (None, 1)                 101

=================================================================
Total params: 15,175,689
Trainable params: 15,175,689
Non-trainable params: 0
_________________________________________________________________
c:\프로그램\ai5\study\keras2\keras76_GAP6_kaggle_cat_dog.py:85: FutureWarning: Passing a negative integer is deprecated in version 1.0 and will not be supported in future version. Instead, use None to not limit the column width.
  pd.set_option('max_colwidth', -1)
                                                          Layer Type Layer Name  Layer Trainable
0  <keras.engine.functional.Functional object at 0x000001FAAB4E9160>  vgg16      True
1  <keras.layers.core.flatten.Flatten object at 0x000001FAAB4E4280>   flatten    True
2  <keras.layers.core.dense.Dense object at 0x000001FAAB651FA0>       dense      True
3  <keras.layers.core.dense.Dense object at 0x000001FAAB658580>       dense_1    True
'''

import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
print(results)
>>>>>>> 70eabacb3fa0ad4089229f1c83ce2c346b0e48a8
