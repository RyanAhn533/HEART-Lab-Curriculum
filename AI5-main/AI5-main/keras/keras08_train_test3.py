import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=123)

print('x_train : ' , x_train)
print('x_test : ' , x_test)
print('y_train : ' , y_train)
print('y_test : ' , y_test)

""" def train_test_split(a, b)
    a = a + b
    return x_train, x_test, y_train, y_test """
