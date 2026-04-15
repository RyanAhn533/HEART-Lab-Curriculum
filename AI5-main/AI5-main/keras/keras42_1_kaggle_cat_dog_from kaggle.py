#https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/data

#배치를 100으로 잡고
#x, y를 추출해서 모델을 맹그러봐
#acc 0.99이상

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Input, MaxPooling2D
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
import time
from sklearn.model_selection import train_test_split


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


val_batch = 10
train_batch = 32

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.4,
        zoom_range=0.3,
        validation_split=0.30,
        horizontal_flip=True,
        )
train_generator = train_datagen.flow_from_directory(
        'C:/프로그램/ai5/_data/kaggle/dogs_vs_cats/train/',
        target_size=(100, 100),
        batch_size=train_datagen,
        class_mode='binary',
        subset = 'training',
        color_mode = 'rgb',
        shuffle = True,
        )
validation_generator = train_datagen.flow_from_directory(
        'C:/프로그램/ai5/_data/kaggle/dogs_vs_cats/train/',
        target_size=(100, 100),
        batch_size=val_batch,
        class_mode='binary',
        subset = 'validation',
        color_mode = 'rgb',
        shuffle= True)


print(train_generator[0][0].shape)
print(train_generator[0][1].shape)

#2. 모델 구성
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Flatten
base_model = keras.applications.Xception(
    weights="imagenet",  # Load weights pre-trained on ImageNet.
    input_shape=(100, 100, 3),
    include_top=False,
)  # Do not include the ImageNet classifier at the top.

# Freeze the base_model
base_model.trainable = False

# Create new model on top
inputs = keras.Input(shape=(100, 100, 3))
# x = data_augmentation(inputs)  # Apply random data augmentation

# Pre-trained Xception weights requires that input be scaled
# from (0, 255) to a range of (-1., +1.), the rescaling layer
# outputs: `(inputs * scale) + offset`
# scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
# x = scale_layer(x)

# The base model contains batchnorm layers. We want to keep them in inference mode
# when we unfreeze the base model for fine-tuning, so we make sure that the
# base_model is running in inference mode here.
x = base_model(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

model.summary()
#3. 컴파일 훈련
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy_score'])

import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')

path1 = './_data/kaggle/dogs_vs_cats/saved_model'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path1, 'k30_', date, '_', filename])
es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=30,
                   restore_best_weights=True)

mcp = ModelCheckpoint(
    monitor= 'val_loss',
    mode = 'auto',  
    verbose=1,
    save_best_only= True,
    filepath = filepath
)

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()],
)

epochs = 20
model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

base_model.trainable = True
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()],
)

epochs = 10
model.fit(train_generator, epochs=epochs, validation_data=validation_generator)