import numpy as py
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import pandas as pd
from tensorflow.keras.datasets import mnist
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, LSTM
from tensorflow.keras.optimizers import Adam

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data() #알아서 데이터 나눠줌
# print(x_train)

##### 스케일링 1-1
x_train = x_train/255.
x_test = x_test/255.

x_train = x_train.reshape(60000, 784, 1)
x_test = x_test.reshape(10000, 784, 1)

print(x_train.shape)

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

y_test = y_test.to_numpy()

#2. 모델 구성
def build_model(drop=0.5, optimizer='adam', activation='relu',
                node1=128, node2=64, node3=32, node4=16, node5=8, lr=0.001):
    inputs = Input(shape=(784, 1), name='inputs')  # 입력 차원 변경
    x = Conv1D(filters=10, kernel_size=2, name='hidden1')(inputs)
    x = Flatten()(x)
    x = Dense(node2, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    x = Dense(node4, activation=activation, name='hidden4')(x)
    x = Dense(node5, activation=activation, name='hidden5')(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)

    optimizer = Adam(learning_rate=lr)  # learning rate 적용
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'], loss='categorical_crossentropy')

    return model

def create_hyperparameter():
    batchs = [32,16,8,1,64]
    optimizers = ['adam', 'rmsprop', 'adadelta']
    dropouts = [0.2, 0.3, 0.4, 0.5]
    lrs = [0.5, 0.1, 0.01, 0.001]
    activations = ['relu', 'elu', 'selu', 'linear']
    node1 = [128, 64, 32, 16]
    node2 = [128, 64, 32, 16]
    node3 = [128, 64, 32, 16]
    node4 = [128, 64, 32, 16]
    node5 = [128, 64, 32, 16, 8]
    return {'batch_size' : batchs,
            'optimizer' : optimizers,
            'drop' : dropouts,
            'activation' : activations,
            'node1' : node1,
            'node2' : node2,
            'node3' : node3,
            'node4' : node4,
            'node5' : node5,
            'lr' : lrs
            }

hyperparameters = create_hyperparameter()
print(hyperparameters)
# {'batch_size': [100, 200, 300, 400, 500], 'optimizer': ['adam', 'rmsprop', 'adadelta'], 'drop': [0.2, 0.3, 0.4, 0.5], 'activation': ['relu', 'elu', 'selu', 'linear'], 
# 'node1': [128, 64, 32, 16], 'node2': [128, 64, 32, 16], 'node3': [128, 64, 32, 16], 'node4': [128, 64, 32, 16], 'node5': [128, 64, 32, 16, 8]}


Keras_model = KerasClassifier(build_fn=build_model, verbose=1)


model = RandomizedSearchCV(Keras_model, hyperparameters, cv=5,
                           n_iter=10,
                        #    n_jobs=-1,
                           verbose=1,
                           )

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    verbose=1,
    patience=3,
    restore_best_weights=True)
import time
start = time.time()

model.fit(x_train, y_train, epochs=5, validation_split=0.2, callbacks=[ReduceLROnPlateau(monitor='val_loss', mode = 'auto', 
                        patience=5, verbose=1, factor=0.8)#running rate * factor)
, es])

end = time.time()

print('time :', round(end - start, 2))
print('model.best_params_ :', model.best_params_)
print('model.best_estimator_ :', model.best_estimator_)
print('model.best_score_ :', model.best_score_)
print('model.score :', model.score(x_test, y_test))

'''
time : 189.63
model.best_params_ : {'optimizer': 'adadelta', 'node5': 8, 'node4': 64, 'node3': 64, 'node2': 128, 'node1': 64, 'lr': 0.001, 'drop': 0.2, 'batch_size': 400, 'activation': 'relu'}
model.best_estimator_ : <keras.wrappers.scikit_learn.KerasClassifier object at 0x000001E5D8A18910>
model.best_score_ : 0.9607166767120361
25/25 [==============================] - 0s 2ms/step - loss: 0.1274 - acc: 0.9665
model.score : 0.9664999842643738
'''



