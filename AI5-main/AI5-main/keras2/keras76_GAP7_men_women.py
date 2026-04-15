<<<<<<< HEAD
<<<<<<< HEAD
from tensorflow.keras.preprocessing.image import load_img # = 이미지 땡겨와
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array # 땡겨온거 수치화
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential


# 데이터 로드
np_path = 'c:/프로그램/ai5/_data/kaggle/Biggest_gender/'
x_train1 = np.load(np_path + 'man_x_train1.npy')
y_train1 = np.load(np_path + 'man_y_train1.npy')
x_test1 = np.load(np_path + 'man_x_test.npy')
y_test1 = np.load(np_path + 'man_y_test.npy')
path = "C:\\프로그램\\ai5\\_data\\image\\me\\me1\\2.jpg"
img = load_img(path, target_size=(200, 200,))
path1 = "C:\\프로그램\\ai5\\_data\\image\\me\\"
arr = img_to_array(img)
img = np.expand_dims(arr, axis=0)
test_datagen = ImageDataGenerator(rescale=1./255)
xy_test = test_datagen.flow_from_directory(
    path1, target_size=(100, 100),
    batch_size=30000, 
    class_mode='binary',
    color_mode='rgb',)

x_train2 = np.load(np_path + 'woman_x_train1.npy')
y_train2 = np.load(np_path + 'woman_y_train2.npy')

# 데이터 결합
x_train = np.concatenate((x_train1, x_train2), axis=0)
y_train = np.concatenate((y_train1, y_train2), axis=0)
x_test = np.concatenate((x_test1, x_train2), axis=0)
y_test = np.concatenate((y_test1, y_train2), axis=0)

# 데이터 증강
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.2,
    shear_range=0.7,
    fill_mode="nearest"
)

# 데이터 증강 및 추가
augment_size = 5000
randidx = np.random.randint(x_train.shape[0], size=augment_size)
x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False
).next()[0]

x_train = np.concatenate((x_train, x_augmented), axis=0)
y_train = np.concatenate((y_train, y_augmented), axis=0)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.9, random_state=42)

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(100, 100, 3))  # input_shape을 이미지 크기와 일치시킴

# 모델 생성
model = Sequential()
model.add(vgg16)          # VGG16을 모델에 추가
#model.add(Flatten())      # Flatten을 추가하여 1차원으로 평탄화
model.add(GlobalAveragePooling2D())
model.add(Dense(100, activation='relu'))  # 완전 연결층 추가
model.add(Dense(1, activation='sigmoid')) # 이진 분류를 위한 출력층

model.summary()  # 모델 구조 출력하여 각 레이어의 출력 형상 확인

import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
print(results)

'''
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
c:\프로그램\ai5\study\keras2\keras76_GAP7_men_women.py:81: FutureWarning: Passing a negative integer is deprecated in version 1.0 and will not be supported in future version. Instead, use None to not limit the column width.
  pd.set_option('max_colwidth', -1)
                                                          Layer Type Layer Name  Layer Trainable
0  <keras.engine.functional.Functional object at 0x00000205C8B930D0>  vgg16      True
1  <keras.layers.core.flatten.Flatten object at 0x00000205C8B985E0>   flatten    True
2  <keras.layers.core.dense.Dense object at 0x00000205C8C10BE0>       dense      True
3  <keras.layers.core.dense.Dense object at 0x00000205C8C188E0>       dense_1    True

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
c:\프로그램\ai5\study\keras2\keras76_GAP7_men_women.py:81: FutureWarning: Passing a negative integer is deprecated in version 1.0 and will not be supported in future version. Instead, use None to not limit the column width.
  pd.set_option('max_colwidth', -1)
                                                                   Layer Type                Layer Name  Layer Trainable
0  <keras.engine.functional.Functional object at 0x000002A59BB65400>           vgg16                     True
1  <keras.layers.pooling.GlobalAveragePooling2D object at 0x000002A59BB684F0>  global_average_pooling2d  True
2  <keras.layers.core.dense.Dense object at 0x000002A59C0FF580>                dense                     True
3  <keras.layers.core.dense.Dense object at 0x000002A59C129BE0>                dense_1                   True

=======
from tensorflow.keras.preprocessing.image import load_img # = 이미지 땡겨와
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array # 땡겨온거 수치화
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential


# 데이터 로드
np_path = 'c:/프로그램/ai5/_data/kaggle/Biggest_gender/'
x_train1 = np.load(np_path + 'man_x_train1.npy')
y_train1 = np.load(np_path + 'man_y_train1.npy')
x_test1 = np.load(np_path + 'man_x_test.npy')
y_test1 = np.load(np_path + 'man_y_test.npy')
path = "C:\\프로그램\\ai5\\_data\\image\\me\\me1\\2.jpg"
img = load_img(path, target_size=(200, 200,))
path1 = "C:\\프로그램\\ai5\\_data\\image\\me\\"
arr = img_to_array(img)
img = np.expand_dims(arr, axis=0)
test_datagen = ImageDataGenerator(rescale=1./255)
xy_test = test_datagen.flow_from_directory(
    path1, target_size=(100, 100),
    batch_size=30000, 
    class_mode='binary',
    color_mode='rgb',)

x_train2 = np.load(np_path + 'woman_x_train1.npy')
y_train2 = np.load(np_path + 'woman_y_train2.npy')

# 데이터 결합
x_train = np.concatenate((x_train1, x_train2), axis=0)
y_train = np.concatenate((y_train1, y_train2), axis=0)
x_test = np.concatenate((x_test1, x_train2), axis=0)
y_test = np.concatenate((y_test1, y_train2), axis=0)

# 데이터 증강
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.2,
    shear_range=0.7,
    fill_mode="nearest"
)

# 데이터 증강 및 추가
augment_size = 5000
randidx = np.random.randint(x_train.shape[0], size=augment_size)
x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False
).next()[0]

x_train = np.concatenate((x_train, x_augmented), axis=0)
y_train = np.concatenate((y_train, y_augmented), axis=0)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.9, random_state=42)

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(100, 100, 3))  # input_shape을 이미지 크기와 일치시킴

# 모델 생성
model = Sequential()
model.add(vgg16)          # VGG16을 모델에 추가
#model.add(Flatten())      # Flatten을 추가하여 1차원으로 평탄화
model.add(GlobalAveragePooling2D())
model.add(Dense(100, activation='relu'))  # 완전 연결층 추가
model.add(Dense(1, activation='sigmoid')) # 이진 분류를 위한 출력층

model.summary()  # 모델 구조 출력하여 각 레이어의 출력 형상 확인

import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
print(results)

'''
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
c:\프로그램\ai5\study\keras2\keras76_GAP7_men_women.py:81: FutureWarning: Passing a negative integer is deprecated in version 1.0 and will not be supported in future version. Instead, use None to not limit the column width.
  pd.set_option('max_colwidth', -1)
                                                          Layer Type Layer Name  Layer Trainable
0  <keras.engine.functional.Functional object at 0x00000205C8B930D0>  vgg16      True
1  <keras.layers.core.flatten.Flatten object at 0x00000205C8B985E0>   flatten    True
2  <keras.layers.core.dense.Dense object at 0x00000205C8C10BE0>       dense      True
3  <keras.layers.core.dense.Dense object at 0x00000205C8C188E0>       dense_1    True

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
c:\프로그램\ai5\study\keras2\keras76_GAP7_men_women.py:81: FutureWarning: Passing a negative integer is deprecated in version 1.0 and will not be supported in future version. Instead, use None to not limit the column width.
  pd.set_option('max_colwidth', -1)
                                                                   Layer Type                Layer Name  Layer Trainable
0  <keras.engine.functional.Functional object at 0x000002A59BB65400>           vgg16                     True
1  <keras.layers.pooling.GlobalAveragePooling2D object at 0x000002A59BB684F0>  global_average_pooling2d  True
2  <keras.layers.core.dense.Dense object at 0x000002A59C0FF580>                dense                     True
3  <keras.layers.core.dense.Dense object at 0x000002A59C129BE0>                dense_1                   True

>>>>>>> cd855f8 (message)
=======
from tensorflow.keras.preprocessing.image import load_img # = 이미지 땡겨와
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array # 땡겨온거 수치화
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential


# 데이터 로드
np_path = 'c:/프로그램/ai5/_data/kaggle/Biggest_gender/'
x_train1 = np.load(np_path + 'man_x_train1.npy')
y_train1 = np.load(np_path + 'man_y_train1.npy')
x_test1 = np.load(np_path + 'man_x_test.npy')
y_test1 = np.load(np_path + 'man_y_test.npy')
path = "C:\\프로그램\\ai5\\_data\\image\\me\\me1\\2.jpg"
img = load_img(path, target_size=(200, 200,))
path1 = "C:\\프로그램\\ai5\\_data\\image\\me\\"
arr = img_to_array(img)
img = np.expand_dims(arr, axis=0)
test_datagen = ImageDataGenerator(rescale=1./255)
xy_test = test_datagen.flow_from_directory(
    path1, target_size=(100, 100),
    batch_size=30000, 
    class_mode='binary',
    color_mode='rgb',)

x_train2 = np.load(np_path + 'woman_x_train1.npy')
y_train2 = np.load(np_path + 'woman_y_train2.npy')

# 데이터 결합
x_train = np.concatenate((x_train1, x_train2), axis=0)
y_train = np.concatenate((y_train1, y_train2), axis=0)
x_test = np.concatenate((x_test1, x_train2), axis=0)
y_test = np.concatenate((y_test1, y_train2), axis=0)

# 데이터 증강
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.2,
    shear_range=0.7,
    fill_mode="nearest"
)

# 데이터 증강 및 추가
augment_size = 5000
randidx = np.random.randint(x_train.shape[0], size=augment_size)
x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False
).next()[0]

x_train = np.concatenate((x_train, x_augmented), axis=0)
y_train = np.concatenate((y_train, y_augmented), axis=0)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.9, random_state=42)

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(100, 100, 3))  # input_shape을 이미지 크기와 일치시킴

# 모델 생성
model = Sequential()
model.add(vgg16)          # VGG16을 모델에 추가
#model.add(Flatten())      # Flatten을 추가하여 1차원으로 평탄화
model.add(GlobalAveragePooling2D())
model.add(Dense(100, activation='relu'))  # 완전 연결층 추가
model.add(Dense(1, activation='sigmoid')) # 이진 분류를 위한 출력층

model.summary()  # 모델 구조 출력하여 각 레이어의 출력 형상 확인

import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
print(results)

'''
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
c:\프로그램\ai5\study\keras2\keras76_GAP7_men_women.py:81: FutureWarning: Passing a negative integer is deprecated in version 1.0 and will not be supported in future version. Instead, use None to not limit the column width.
  pd.set_option('max_colwidth', -1)
                                                          Layer Type Layer Name  Layer Trainable
0  <keras.engine.functional.Functional object at 0x00000205C8B930D0>  vgg16      True
1  <keras.layers.core.flatten.Flatten object at 0x00000205C8B985E0>   flatten    True
2  <keras.layers.core.dense.Dense object at 0x00000205C8C10BE0>       dense      True
3  <keras.layers.core.dense.Dense object at 0x00000205C8C188E0>       dense_1    True

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
c:\프로그램\ai5\study\keras2\keras76_GAP7_men_women.py:81: FutureWarning: Passing a negative integer is deprecated in version 1.0 and will not be supported in future version. Instead, use None to not limit the column width.
  pd.set_option('max_colwidth', -1)
                                                                   Layer Type                Layer Name  Layer Trainable
0  <keras.engine.functional.Functional object at 0x000002A59BB65400>           vgg16                     True
1  <keras.layers.pooling.GlobalAveragePooling2D object at 0x000002A59BB684F0>  global_average_pooling2d  True
2  <keras.layers.core.dense.Dense object at 0x000002A59C0FF580>                dense                     True
3  <keras.layers.core.dense.Dense object at 0x000002A59C129BE0>                dense_1                   True

>>>>>>> 70eabacb3fa0ad4089229f1c83ce2c346b0e48a8
'''