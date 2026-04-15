<<<<<<< HEAD
<<<<<<< HEAD
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten,LSTM, Dropout, Input, MaxPooling2D
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
import time
from sklearn.model_selection import train_test_split
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


train_datagen = ImageDataGenerator(
    rescale=1./255,)
path_train = 'C:/프로그램/ai5/_data/image/rps/'


xy_train = train_datagen.flow_from_directory(
    path_train, target_size=(100, 100), 
    batch_size=20000, 
class_mode='sparse',

color_mode='rgb',
shuffle=True)


x_train,x_test, y_train, y_test = train_test_split(xy_train[0][0], xy_train[0][1], train_size=0.7, random_state=3)

print(x_train.shape, x_test.shape)
#(1764, 100, 100, 3) (756, 100, 100, 3)
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(100, 100, 3))  # input_shape을 이미지 크기와 일치시킴

# 모델 생성
model = Sequential()
model.add(vgg16)          # VGG16을 모델에 추가
#model.add(Flatten())      # Flatten을 추가하여 1차원으로 평탄화
model.add(GlobalAveragePooling2D())
model.add(Dense(100, activation='relu'))  # 완전 연결층 추가
model.add(Dense(1, activation='sigmoid')) # 이진 분류를 위한 출력층

model.summary()  # 모델 구조 출력하여 각 레이어의 출력 형상 확인

'''
Flatten
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
0  <keras.engine.functional.Functional object at 0x0000019930666A30>  vgg16      True
1  <keras.layers.core.flatten.Flatten object at 0x0000019930674A00>   flatten    True
2  <keras.layers.core.dense.Dense object at 0x00000199306972E0>       dense      True
3  <keras.layers.core.dense.Dense object at 0x0000019936DB4A30>       dense_1    True

GAP
________________________________________________________
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
0  <keras.engine.functional.Functional object at 0x000002B0E8986CD0>           vgg16                     True
1  <keras.layers.pooling.GlobalAveragePooling2D object at 0x000002B0E89949D0>  global_average_pooling2d  True
2  <keras.layers.core.dense.Dense object at 0x000002B0E89B52B0>                dense                     True
3  <keras.layers.core.dense.Dense object at 0x000002B0E8B02700>                dense_1                   True
'''

import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
print(results)
=======
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten,LSTM, Dropout, Input, MaxPooling2D
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
import time
from sklearn.model_selection import train_test_split
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


train_datagen = ImageDataGenerator(
    rescale=1./255,)
path_train = 'C:/프로그램/ai5/_data/image/rps/'


xy_train = train_datagen.flow_from_directory(
    path_train, target_size=(100, 100), 
    batch_size=20000, 
class_mode='sparse',

color_mode='rgb',
shuffle=True)


x_train,x_test, y_train, y_test = train_test_split(xy_train[0][0], xy_train[0][1], train_size=0.7, random_state=3)

print(x_train.shape, x_test.shape)
#(1764, 100, 100, 3) (756, 100, 100, 3)
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(100, 100, 3))  # input_shape을 이미지 크기와 일치시킴

# 모델 생성
model = Sequential()
model.add(vgg16)          # VGG16을 모델에 추가
#model.add(Flatten())      # Flatten을 추가하여 1차원으로 평탄화
model.add(GlobalAveragePooling2D())
model.add(Dense(100, activation='relu'))  # 완전 연결층 추가
model.add(Dense(1, activation='sigmoid')) # 이진 분류를 위한 출력층

model.summary()  # 모델 구조 출력하여 각 레이어의 출력 형상 확인

'''
Flatten
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
0  <keras.engine.functional.Functional object at 0x0000019930666A30>  vgg16      True
1  <keras.layers.core.flatten.Flatten object at 0x0000019930674A00>   flatten    True
2  <keras.layers.core.dense.Dense object at 0x00000199306972E0>       dense      True
3  <keras.layers.core.dense.Dense object at 0x0000019936DB4A30>       dense_1    True

GAP
________________________________________________________
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
0  <keras.engine.functional.Functional object at 0x000002B0E8986CD0>           vgg16                     True
1  <keras.layers.pooling.GlobalAveragePooling2D object at 0x000002B0E89949D0>  global_average_pooling2d  True
2  <keras.layers.core.dense.Dense object at 0x000002B0E89B52B0>                dense                     True
3  <keras.layers.core.dense.Dense object at 0x000002B0E8B02700>                dense_1                   True
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
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten,LSTM, Dropout, Input, MaxPooling2D
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
import time
from sklearn.model_selection import train_test_split
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


train_datagen = ImageDataGenerator(
    rescale=1./255,)
path_train = 'C:/프로그램/ai5/_data/image/rps/'


xy_train = train_datagen.flow_from_directory(
    path_train, target_size=(100, 100), 
    batch_size=20000, 
class_mode='sparse',

color_mode='rgb',
shuffle=True)


x_train,x_test, y_train, y_test = train_test_split(xy_train[0][0], xy_train[0][1], train_size=0.7, random_state=3)

print(x_train.shape, x_test.shape)
#(1764, 100, 100, 3) (756, 100, 100, 3)
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(100, 100, 3))  # input_shape을 이미지 크기와 일치시킴

# 모델 생성
model = Sequential()
model.add(vgg16)          # VGG16을 모델에 추가
#model.add(Flatten())      # Flatten을 추가하여 1차원으로 평탄화
model.add(GlobalAveragePooling2D())
model.add(Dense(100, activation='relu'))  # 완전 연결층 추가
model.add(Dense(1, activation='sigmoid')) # 이진 분류를 위한 출력층

model.summary()  # 모델 구조 출력하여 각 레이어의 출력 형상 확인

'''
Flatten
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
0  <keras.engine.functional.Functional object at 0x0000019930666A30>  vgg16      True
1  <keras.layers.core.flatten.Flatten object at 0x0000019930674A00>   flatten    True
2  <keras.layers.core.dense.Dense object at 0x00000199306972E0>       dense      True
3  <keras.layers.core.dense.Dense object at 0x0000019936DB4A30>       dense_1    True

GAP
________________________________________________________
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
0  <keras.engine.functional.Functional object at 0x000002B0E8986CD0>           vgg16                     True
1  <keras.layers.pooling.GlobalAveragePooling2D object at 0x000002B0E89949D0>  global_average_pooling2d  True
2  <keras.layers.core.dense.Dense object at 0x000002B0E89B52B0>                dense                     True
3  <keras.layers.core.dense.Dense object at 0x000002B0E8B02700>                dense_1                   True
'''

import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
print(results)
>>>>>>> 70eabacb3fa0ad4089229f1c83ce2c346b0e48a8
