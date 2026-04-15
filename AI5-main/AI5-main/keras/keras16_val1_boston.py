from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import sklearn as sk
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1.데이터
dataset = load_boston()

x = dataset.data
y = dataset.target


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7, shuffle=True, 
                                                    random_state=3)
x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.2, shuffle=True, random_state=3)
#2. 모델 input dim 13 / output 1
model = Sequential()
model.add(Dense(1, input_dim=13))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))


#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=10, verbose=0, validation_data=(x_val, y_val))

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print("로스 : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2) # 1000 + 512    10000 + 512 -  1000 102  0.68
