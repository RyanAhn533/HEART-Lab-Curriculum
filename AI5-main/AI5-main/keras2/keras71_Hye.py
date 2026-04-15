import numpy as py
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import warnings
warnings.filterwarnings('ignore')

#1. 데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, stratify=y, shuffle=True, random_state=336)

print(x_train.shape, y_train.shape)     # (455, 30) (455,)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성
def build_model(drop=0.5, optimizer=Adam(learning_rate=rlr), activation='relu',
                node1=128, node2=64, node3=32, node4=16, node5=8, lr=0.001):
    inputs = Input(shape=(30, ), name='inputs')
    x = Dense(node1, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    x = Dense(node4, activation=activation, name='hidden4')(x)
    x = Dense(node5, activation=activation, name='hidden5')(x)
    outputs = Dense(1, activation='sigmoid', name='outputs')(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=optimizer, metrics=['accuracy'], loss='mse')
    
    return model

def create_hyperparameter():
    batchs = [32, 16, 8, 1, 64]
    optimizers = ['Adam', 'rmsprop', 'adadelta']
    lr = [0.01, 0.005, 0.001, 0.0005]
    dropouts = [0.2, 0.3, 0.4, 0.5]
    activations = ['relu', 'elu', 'selu', 'linear']
    node1 = [128, 64, 32, 16]
    node2 = [128, 64, 32, 16]
    node3 = [128, 64, 32, 16]
    node4 = [128, 64, 32, 16]
    node5 = [128, 64, 32, 16, 8]
    return {'batch_size' : batchs,
            'optimizer' : optimizers,
            'lr' : lr,
            'drop' : dropouts,
            'activation' : activations,
            'node1' : node1,
            'node2' : node2,
            'node3' : node3,
            'node4' : node4,
            'node5' : node5,
            }

hyperparameters = create_hyperparameter()
print(hyperparameters)
# a

from sklearn.model_selection import RandomizedSearchCV

# sklearn.utils._param_validation.InvalidParameterError: The 'estimator' parameter of RandomizedSearchCV must be an object implementing 'fit'. Got <function build_model at 0x0000022B3EE6FA60> instead.
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

Keras_model = KerasClassifier(build_fn=build_model, verbose=1)

model = RandomizedSearchCV(Keras_model, hyperparameters, cv=3,
                           n_iter=2,
                        #    n_jobs=-1,
                           verbose=1,
                           )

########## mcp 세이브 파일명 만들기 시작 ##########
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

path_w = './_save/keras71/'
filename = '{epoch:04d}_valloss_{val_loss:.4f}.hdf5'
filepath = "".join([path_w, 'k71_06_date_', date, '_epo_', filename])

########## mcp 세이브 파일명 만들기 끝 ##########

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    verbose = 1,
    save_best_only = True,
    filepath = filepath
)

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 24,
    restore_best_weights = True
)

rlr = ReduceLROnPlateau(
    monitor = 'val_loss',
    mode = 'auto',
    patience = 8,
    verbose = 1,
    factor = 0.8
)

import time
start = time.time()
model.fit(x_train, y_train, epochs=5, validation_split = 0.25, callbacks = [es, mcp], verbose=0)
end = time.time()

print('model.best_params_ :', model.best_params_)
print('model.best_estimator_ :', model.best_estimator_)
print('model.best_score_ :', model.best_score_)
print('model.score :', model.score(x_test, y_test))
print('time :', round(end - start, 2))

from sklearn.metrics import accuracy_score, r2_score
y_predict = model.predict(x_test)
print('acc :',accuracy_score(x_test, y_predict))

'''


'''



