import numpy as py
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

#1. 데이터
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=336)

print(x_train.shape, y_train.shape)     # (353, 10) (353,)

#2. 모델 구성
def build_model(drop=0.5, optimizer='adam', activation='relu',
                node1=128, node2=64, node3=32, node4=16, node5=8, lr=0.5):
    inputs = Input(shape=(10, ), name='inputs')
    x = Dense(node1, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    x = Dense(node4, activation=activation, name='hidden4')(x)
    x = Dense(node5, activation=activation, name='hidden5')(x)
    outputs = Dense(1, activation='linear', name='outputs')(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=optimizer, metrics=['mae'], loss='mse')


    return model

def create_hyperparameter():
    batchs = [100, 200, 300, 400, 500]
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


Keras_model = KerasRegressor(build_fn=build_model, verbose=1)


model = RandomizedSearchCV(Keras_model, hyperparameters, cv=5,
                           n_iter=10,
                        #    n_jobs=-1,
                           verbose=1,
                           )

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    verbose=1,
    patience=20,
    restore_best_weights=True)

import time
start = time.time()

model.fit(x_train, y_train, epochs=100, validation_split=0.2, callbacks=[ReduceLROnPlateau(monitor='val_loss', mode = 'auto', 
                        patience=5, verbose=1, factor=0.8)#running rate * factor)
, es])

end = time.time()

print('time :', round(end - start, 2))
print('model.best_params_ :', model.best_params_)
print('model.best_estimator_ :', model.best_estimator_)
print('model.best_score_ :', model.best_score_)
print('model.score :', model.score(x_test, y_test))

'''
sklearn.utils._param_validation.InvalidParameterError: The 'estimator' parameter of RandomizedSearchCV must be an object implementing 'fit'. Got <function build_model at 0x0000022B3EE6FA60> instead.

'''



