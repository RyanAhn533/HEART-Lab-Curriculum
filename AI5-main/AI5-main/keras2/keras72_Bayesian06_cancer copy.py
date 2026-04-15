#DNNimport numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, r2_score

from bayes_opt import BayesianOptimization

import time

#1. 데이터
x, y = load_diabetes(return_X_y=True)

# y = pd.get_dummies(y)
# y = y.reshape(-1,1)
x = x/255.

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=336, train_size=0.8, 
                                                    # stratify=y
                                                    )


print(x_train.shape, y_train.shape) # (353, 10) (353,)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
lebal  = LabelEncoder()

#2. 모델
def build_model(drop=0.5, optimizer='adam', activation='relu',
                node1=128, node2=64, node3=32, node4=16, node5=8, lr=0.001):
    activation = lebal.inverse_transform([int(activation)])[0]
    inputs = Input(shape=(x_train[1],), name='inputs')
    x = Dense(node1, activation=
              
              
              
              activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    x = Dense(node4, activation=activation, name='hidden4')(x)
    x = Dense(node5, activation=activation, name='hidden5')(x)
    outputs = Dense(1, activation='linear', name='outputs')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=le.inverse_transform([int(optimizer)])[0],
                  metrics=['mae'], loss='mse')
    
    model.fit(x_train, y_train, epochs=100, 
            #   callbacks = [mcp, es, rlr],
              validation_split = 0.1,
            #   batch_size=batchs,
              verbose=0,
              )
    
    y_pre = model.predict(x_test)
    
    result = r2_score(y_test, y_pre)
    
    return result     


def create_hyperparameter():
    # batchs = (8, 64)
    optimizers = ['adam', 'rmsprop', 'adadelta']
    optimizers = (0, max(le.fit_transform(optimizers)))
    dropouts = (0.2, 0.5)
    activations = ['relu', 'elu', 'selu', 'linear']
    activations = (0, max(lebal.fit_transform(activations)))
    node1 = (16, 128)
    node2 = (16, 128)
    node3 = (16, 128)
    node4 = (16, 128)
    node5 = (16, 128)
    return {
        # 'batch_size' : batchs,
        'optimizer' : optimizers,
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
# {'batch_size': ([100, 200, 300, 400, 500],), 'optimizer': (['adam', 'rmsprop', 'adadelta'],), 'drop': ([0.2, 0.3, 0.4, 0.5],), 'activation': (['relu', 'elu', 'selu', 'linear'],), 'node1': [128, 64, 32, 16], 'node2': [128, 64, 32, 16], 'node3': [128, 64, 32, 16], 'node4': [128, 64, 32, 16], 'node5': [128, 64, 32, 16, 8]}

from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

keras_model = KerasRegressor(build_fn=build_model, verbose=1, 
                             )

bay = BayesianOptimization(
    f=build_model,
    pbounds=hyperparameters,
    random_state=333    
)

n_iter = 100
st = time.time()
bay.maximize(init_points=5, n_iter=n_iter)  # maximize 가 fit이라고 생각
et = time.time()

print(bay.max)
print(n_iter, '번 걸린 시간 :', round(et-st, 2), '초')

'''
| 95        | 0.00648   | 0.0857    | 0.2118    | 62.47     | 103.6     | 111.5     | 61.55     | 39.65     | 0.9803    |
| 96        | 0.001798  | 0.2583    | 0.4061    | 71.91     | 25.73     | 101.8     | 60.72     | 46.02     | 0.9851    |
| 97        | 0.0004899 | 2.231     | 0.2211    | 47.69     | 28.97     | 39.44     | 98.28     | 104.2     | 0.5998    |
| 98        | 0.05152   | 1.418     | 0.3722    | 114.3     | 124.3     | 68.42     | 87.74     | 39.04     | 1.999     |
| 99        | 0.1405    | 1.418     | 0.3722    | 114.3     | 124.3     | 68.42     | 87.74     | 39.04     | 1.999     |
| 100       | 0.005514  | 2.193     | 0.2544    | 46.36     | 118.0     | 108.0     | 20.63     | 76.01     | 1.746     |
| 101       | 0.01581   | 1.395     | 0.2611    | 18.68     | 103.2     | 91.0      | 127.9     | 102.0     | 0.4347    |
| 102       | 0.0008406 | 0.5002    | 0.4574    | 127.9     | 46.12     | 69.61     | 51.84     | 112.6     | 0.324     |
| 103       | 0.01885   | 2.208     | 0.4314    | 66.46     | 104.0     | 89.28     | 122.4     | 127.8     | 1.258     |
| 104       | 0.1452    | 1.418     | 0.3722    | 114.3     | 124.3     | 68.42     | 87.74     | 39.04     | 1.999     |
| 105       | 0.1232    | 0.6161    | 0.2163    | 112.7     | 109.2     | 60.44     | 117.9     | 119.9     | 1.621     |
=========================================================================================================================
{'target': 0.15491701552545578, 'params': {'activation': 1.4180788519424474, 'drop': 0.3721917258606218, 'node1': 114.28006522828022, 'node2': 124.27246201765578, 'node3': 68.41619925550617, 'node4': 87.73846011886435, 'node5': 39.04149475903363, 'optimizer': 1.99918896988167}}
100 번 걸린 시간 : 906.39 초
'''