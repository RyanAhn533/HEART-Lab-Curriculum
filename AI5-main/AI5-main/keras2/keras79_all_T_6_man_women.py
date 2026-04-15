<<<<<<< HEAD
<<<<<<< HEAD
from tensorflow.keras.applications import VGG16, Xception, ResNet50, ResNet101, InceptionV3
from tensorflow.keras.applications import InceptionResNetV2, DenseNet121, MobileNetV2
from tensorflow.keras.applications import NASNetMobile, EfficientNetB0
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder

model_list = [VGG16(include_top=False, input_shape=(32, 32, 3)),
              ResNet50(include_top=False, input_shape=(32, 32, 3)),
              ResNet101(include_top=False, input_shape=(32, 32, 3)),
              DenseNet121(include_top=False, input_shape=(32, 32, 3)),
              #InceptionV3(include_top=False, input_shape=(32, 32, 3)),
              #InceptionResNetV2(include_top=False, input_shape=(32,32,3)),
              MobileNetV2(include_top=False, input_shape=(32,32,3)),
              #NASNetMobile(include_top=False, input_shape=(32, 32, 3)),
              EfficientNetB0(include_top=False, input_shape=(32,32,3)),
              #Xception(include_top=False, input_shape=(32,32,3))
              ]

################# GAP 쓰기, 기존꺼 최고 꺼랑 성능 비교 #############

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
import tensorflow as tf
import time
from sklearn.metrics import r2_score, accuracy_score

tf.random.set_seed(333)
np.random.seed(333)
print(tf.__version__)   # 2.7.4

from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10

for model_i in model_list:
  model_i.trainable = False
  
  model = Sequential()
  model.add(model_i)
  #model.add(Flatten())
  model.add(GlobalAveragePooling2D())
  model.add(Dense(100))
  model.add(Dense(100))
  model.add(Dense(10, activation='softmax'))
  
  #model.summary
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()
  
  print(x_train.shape, y_train.shape)
  print(x_test.shape, y_test.shape)
  '''
  (50000, 32, 32, 3) (50000, 1)
(10000, 32, 32, 3) (10000, 1)
(50000, 32, 32, 3) (50000, 1)
(10000, 32, 32, 3) (10000, 1)
(50000, 32, 32, 3) (50000, 1)
(10000, 32, 32, 3) (10000, 1)
(50000, 32, 32, 3) (50000, 1)
(10000, 32, 32, 3) (10000, 1)
(50000, 32, 32, 3) (50000, 1)
(10000, 32, 32, 3) (10000, 1)
(50000, 32, 32, 3) (50000, 1)
(10000, 32, 32, 3) (10000, 1)
  '''
  x_train = x_train/255.      # 0~1 사이 값으로 바뀜
  x_test = x_test/255.


  model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
  
  from tensorflow.keras.callbacks import EarlyStopping
  es = EarlyStopping(monitor='val_loss', mode='min',
                     patience=10, verbose=0,
                     restore_best_weights=True)
  
  start = time.time()
  hist = model.fit(x_train, y_train, epochs=1000, batch_size=128,
                   verbose=0,
                   validation_split=0.2,
                   callbacks=[es],)
  end = time.time()
  
  #4. 평가, 예측
  loss = model.evaluate(x_test, y_test, verbose=1)
  print("-----------------------")
  print("모델명 :", model_i.name, 'loss :', loss[0], 'acc :', round(loss[1],2))
  
  # print('loss :', loss[0])
  # print('acc :', round(loss[1],2))

  y_pre = model.predict(x_test)

=======
from tensorflow.keras.applications import VGG16, Xception, ResNet50, ResNet101, InceptionV3
from tensorflow.keras.applications import InceptionResNetV2, DenseNet121, MobileNetV2
from tensorflow.keras.applications import NASNetMobile, EfficientNetB0
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder

model_list = [VGG16(include_top=False, input_shape=(32, 32, 3)),
              ResNet50(include_top=False, input_shape=(32, 32, 3)),
              ResNet101(include_top=False, input_shape=(32, 32, 3)),
              DenseNet121(include_top=False, input_shape=(32, 32, 3)),
              #InceptionV3(include_top=False, input_shape=(32, 32, 3)),
              #InceptionResNetV2(include_top=False, input_shape=(32,32,3)),
              MobileNetV2(include_top=False, input_shape=(32,32,3)),
              #NASNetMobile(include_top=False, input_shape=(32, 32, 3)),
              EfficientNetB0(include_top=False, input_shape=(32,32,3)),
              #Xception(include_top=False, input_shape=(32,32,3))
              ]

################# GAP 쓰기, 기존꺼 최고 꺼랑 성능 비교 #############

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
import tensorflow as tf
import time
from sklearn.metrics import r2_score, accuracy_score

tf.random.set_seed(333)
np.random.seed(333)
print(tf.__version__)   # 2.7.4

from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10

for model_i in model_list:
  model_i.trainable = False
  
  model = Sequential()
  model.add(model_i)
  #model.add(Flatten())
  model.add(GlobalAveragePooling2D())
  model.add(Dense(100))
  model.add(Dense(100))
  model.add(Dense(10, activation='softmax'))
  
  #model.summary
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()
  
  print(x_train.shape, y_train.shape)
  print(x_test.shape, y_test.shape)
  '''
  (50000, 32, 32, 3) (50000, 1)
(10000, 32, 32, 3) (10000, 1)
(50000, 32, 32, 3) (50000, 1)
(10000, 32, 32, 3) (10000, 1)
(50000, 32, 32, 3) (50000, 1)
(10000, 32, 32, 3) (10000, 1)
(50000, 32, 32, 3) (50000, 1)
(10000, 32, 32, 3) (10000, 1)
(50000, 32, 32, 3) (50000, 1)
(10000, 32, 32, 3) (10000, 1)
(50000, 32, 32, 3) (50000, 1)
(10000, 32, 32, 3) (10000, 1)
  '''
  x_train = x_train/255.      # 0~1 사이 값으로 바뀜
  x_test = x_test/255.


  model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
  
  from tensorflow.keras.callbacks import EarlyStopping
  es = EarlyStopping(monitor='val_loss', mode='min',
                     patience=10, verbose=0,
                     restore_best_weights=True)
  
  start = time.time()
  hist = model.fit(x_train, y_train, epochs=1000, batch_size=128,
                   verbose=0,
                   validation_split=0.2,
                   callbacks=[es],)
  end = time.time()
  
  #4. 평가, 예측
  loss = model.evaluate(x_test, y_test, verbose=1)
  print("-----------------------")
  print("모델명 :", model_i.name, 'loss :', loss[0], 'acc :', round(loss[1],2))
  
  # print('loss :', loss[0])
  # print('acc :', round(loss[1],2))

  y_pre = model.predict(x_test)

>>>>>>> cd855f8 (message)
=======
from tensorflow.keras.applications import VGG16, Xception, ResNet50, ResNet101, InceptionV3
from tensorflow.keras.applications import InceptionResNetV2, DenseNet121, MobileNetV2
from tensorflow.keras.applications import NASNetMobile, EfficientNetB0
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder

model_list = [VGG16(include_top=False, input_shape=(32, 32, 3)),
              ResNet50(include_top=False, input_shape=(32, 32, 3)),
              ResNet101(include_top=False, input_shape=(32, 32, 3)),
              DenseNet121(include_top=False, input_shape=(32, 32, 3)),
              #InceptionV3(include_top=False, input_shape=(32, 32, 3)),
              #InceptionResNetV2(include_top=False, input_shape=(32,32,3)),
              MobileNetV2(include_top=False, input_shape=(32,32,3)),
              #NASNetMobile(include_top=False, input_shape=(32, 32, 3)),
              EfficientNetB0(include_top=False, input_shape=(32,32,3)),
              #Xception(include_top=False, input_shape=(32,32,3))
              ]

################# GAP 쓰기, 기존꺼 최고 꺼랑 성능 비교 #############

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
import tensorflow as tf
import time
from sklearn.metrics import r2_score, accuracy_score

tf.random.set_seed(333)
np.random.seed(333)
print(tf.__version__)   # 2.7.4

from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10

for model_i in model_list:
  model_i.trainable = False
  
  model = Sequential()
  model.add(model_i)
  #model.add(Flatten())
  model.add(GlobalAveragePooling2D())
  model.add(Dense(100))
  model.add(Dense(100))
  model.add(Dense(10, activation='softmax'))
  
  #model.summary
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()
  
  print(x_train.shape, y_train.shape)
  print(x_test.shape, y_test.shape)
  '''
  (50000, 32, 32, 3) (50000, 1)
(10000, 32, 32, 3) (10000, 1)
(50000, 32, 32, 3) (50000, 1)
(10000, 32, 32, 3) (10000, 1)
(50000, 32, 32, 3) (50000, 1)
(10000, 32, 32, 3) (10000, 1)
(50000, 32, 32, 3) (50000, 1)
(10000, 32, 32, 3) (10000, 1)
(50000, 32, 32, 3) (50000, 1)
(10000, 32, 32, 3) (10000, 1)
(50000, 32, 32, 3) (50000, 1)
(10000, 32, 32, 3) (10000, 1)
  '''
  x_train = x_train/255.      # 0~1 사이 값으로 바뀜
  x_test = x_test/255.


  model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
  
  from tensorflow.keras.callbacks import EarlyStopping
  es = EarlyStopping(monitor='val_loss', mode='min',
                     patience=10, verbose=0,
                     restore_best_weights=True)
  
  start = time.time()
  hist = model.fit(x_train, y_train, epochs=1000, batch_size=128,
                   verbose=0,
                   validation_split=0.2,
                   callbacks=[es],)
  end = time.time()
  
  #4. 평가, 예측
  loss = model.evaluate(x_test, y_test, verbose=1)
  print("-----------------------")
  print("모델명 :", model_i.name, 'loss :', loss[0], 'acc :', round(loss[1],2))
  
  # print('loss :', loss[0])
  # print('acc :', round(loss[1],2))

  y_pre = model.predict(x_test)

>>>>>>> 70eabacb3fa0ad4089229f1c83ce2c346b0e48a8
  y_pre = np.argmax(y_pre, axis=1).reshape(-1,1)