from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. 데이터
dataset = load_diabetes()
x = dataset.data
y = dataset.target

# 스케일링 적용
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

# KFold 설정
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)

# 2. 모델 훈련 및 평가
r2_scores = []
losses = []

for train_index, test_index in kfold.split(x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model = Sequential()
    model.add(Dense(128, input_dim=10, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1))

    model.compile(
        loss='mse',
        optimizer='adam',
        metrics=['acc'])

    es = EarlyStopping(monitor='val_loss', mode='min', patience=20, 
                       restore_best_weights=True)
    
    mcp = ModelCheckpoint(
        monitor='val_loss',
        mode='auto',
        verbose=1,
        save_best_only=True, 
        filepath=("./_save/keras32/keras32_dropout.h1"))

    model.fit(x_train, y_train, epochs=200, batch_size=128, 
              verbose=0, validation_split=0.2, callbacks=[es])

    # 평가 및 예측
    loss = model.evaluate(x_test, y_test)
    losses.append(loss)
    y_predict = model.predict(x_test)
    r2 = r2_score(y_test, y_predict)
    r2_scores.append(r2)
    
    print("로스는 ?", loss)
    print("r2스코어는? ", r2)

# KFold의 평균 결과 출력
print(f"\n최종 평균 로스: {np.mean(losses)}")
print(f"최종 평균 r2스코어: {np.mean(r2_scores)}")
