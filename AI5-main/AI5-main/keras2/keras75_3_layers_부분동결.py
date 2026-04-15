<<<<<<< HEAD
<<<<<<< HEAD
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.applications import VGG16

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

#include_top=False -> fully connetec 삭제한다. = VGG16 16개 중 3개가 fully conneted layer - 삭제
# Dense layer가 두개 / 첫 시작 bias + wegiht = 2 -> 13 * 2 + dense 2개 = bias 2 개 + weights 2개
# input_shape 고정
model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

#1. 전체동결
# model.trainable = False # 가중치 동결/ 두개는 훈련을 시켜저ㅜ야함

# #2. 전체 동결
# for layer in model.layers:        #이터레이터
#     layer.trainable = False
    
#3. 부분동결
print(model.layers[2])
#[<keras.engine.functional.Functional object at 0x000001CEA6A8E430>, 
# <keras.layers.core.flatten.Flatten object at 0x000001CEA6A95940>,
# <keras.layers.core.dense.Dense object at 0x000001CEA6ADA5B0>, 
# <keras.layers.core.dense.Dense object at 0x000001CEA6B0DA00>]

model.layers[1].trainable = False
model.summary()

import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
print(results)

=======
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.applications import VGG16

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

#include_top=False -> fully connetec 삭제한다. = VGG16 16개 중 3개가 fully conneted layer - 삭제
# Dense layer가 두개 / 첫 시작 bias + wegiht = 2 -> 13 * 2 + dense 2개 = bias 2 개 + weights 2개
# input_shape 고정
model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

#1. 전체동결
# model.trainable = False # 가중치 동결/ 두개는 훈련을 시켜저ㅜ야함

# #2. 전체 동결
# for layer in model.layers:        #이터레이터
#     layer.trainable = False
    
#3. 부분동결
print(model.layers[2])
#[<keras.engine.functional.Functional object at 0x000001CEA6A8E430>, 
# <keras.layers.core.flatten.Flatten object at 0x000001CEA6A95940>,
# <keras.layers.core.dense.Dense object at 0x000001CEA6ADA5B0>, 
# <keras.layers.core.dense.Dense object at 0x000001CEA6B0DA00>]

model.layers[1].trainable = False
model.summary()

import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
print(results)

>>>>>>> cd855f8 (message)
=======
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.applications import VGG16

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

#include_top=False -> fully connetec 삭제한다. = VGG16 16개 중 3개가 fully conneted layer - 삭제
# Dense layer가 두개 / 첫 시작 bias + wegiht = 2 -> 13 * 2 + dense 2개 = bias 2 개 + weights 2개
# input_shape 고정
model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

#1. 전체동결
# model.trainable = False # 가중치 동결/ 두개는 훈련을 시켜저ㅜ야함

# #2. 전체 동결
# for layer in model.layers:        #이터레이터
#     layer.trainable = False
    
#3. 부분동결
print(model.layers[2])
#[<keras.engine.functional.Functional object at 0x000001CEA6A8E430>, 
# <keras.layers.core.flatten.Flatten object at 0x000001CEA6A95940>,
# <keras.layers.core.dense.Dense object at 0x000001CEA6ADA5B0>, 
# <keras.layers.core.dense.Dense object at 0x000001CEA6B0DA00>]

model.layers[1].trainable = False
model.summary()

import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
print(results)

>>>>>>> 70eabacb3fa0ad4089229f1c83ce2c346b0e48a8
