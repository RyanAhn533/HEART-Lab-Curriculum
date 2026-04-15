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

vgg16.trainable = False

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

#model.trainable = False # 가중치 동결/ 두개는 훈련을 시켜저ㅜ야함
model.summary()

print(len(model.weights))
print(len(model.trainable_weights))
#30
#0   why?

import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
print(results)

"""
                        Trainable:True  // model = False  // VGG false
trainable = True          30            //      0         //   30
len(model.weights)        30            //      0         //   
    
=======
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.applications import VGG16

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

#include_top=False -> fully connetec 삭제한다. = VGG16 16개 중 3개가 fully conneted layer - 삭제
# Dense layer가 두개 / 첫 시작 bias + wegiht = 2 -> 13 * 2 + dense 2개 = bias 2 개 + weights 2개
# input_shape 고정

vgg16.trainable = False

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

#model.trainable = False # 가중치 동결/ 두개는 훈련을 시켜저ㅜ야함
model.summary()

print(len(model.weights))
print(len(model.trainable_weights))
#30
#0   why?

import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
print(results)

"""
                        Trainable:True  // model = False  // VGG false
trainable = True          30            //      0         //   30
len(model.weights)        30            //      0         //   
    
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

vgg16.trainable = False

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

#model.trainable = False # 가중치 동결/ 두개는 훈련을 시켜저ㅜ야함
model.summary()

print(len(model.weights))
print(len(model.trainable_weights))
#30
#0   why?

import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
print(results)

"""
                        Trainable:True  // model = False  // VGG false
trainable = True          30            //      0         //   30
len(model.weights)        30            //      0         //   
    
>>>>>>> 70eabacb3fa0ad4089229f1c83ce2c346b0e48a8
"""