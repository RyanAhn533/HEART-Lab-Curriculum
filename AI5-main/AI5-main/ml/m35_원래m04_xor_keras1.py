import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

#1. 데이터
x_data = np.array([[0,0], [0,1], [1,0], [1,1]])
y_data = np.array([0,1,1,0])
print(x_data.shape, y_data.shape)

# 모델 생성
model = Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid'))

#3. 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, batch_size=1, epochs=100)

#4. 평가
loss, acc = model.evaluate(x_data, y_data)
print('Model Loss:', loss)
print('Model Accuracy:', acc)

# 예측
y_predict = model.predict(x_data).reshape(-1,).astype(int)

# 정확도 비교
acc2 = accuracy_score(y_data, y_predict)
print('Accuracy score:', acc2)

print("===================================================")
print('실제 값:', y_data)
print('예측 값:', y_predict)
