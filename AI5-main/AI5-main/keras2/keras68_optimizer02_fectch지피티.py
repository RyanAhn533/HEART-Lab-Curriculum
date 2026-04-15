import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
import time

# 1. 데이터 로드 및 전처리
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

scaler = StandardScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=3)

# 2. PCA 적용
pca = PCA(n_components=54)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

# PCA 설명력 출력
evr = pca.explained_variance_ratio_
evr_cumsum = np.cumsum(evr)
print(f'0.95 이상: {np.argmax(evr_cumsum >= 0.95) + 1}')
print(f'0.99 이상: {np.argmax(evr_cumsum >= 0.99) + 1}')
print(f'0.999 이상: {np.argmax(evr_cumsum >= 0.999) + 1}')
print(f'1.0 이상: {np.argmax(evr_cumsum >= 1) + 1}')

# 3. 모델 구성
model = Sequential([
    Dense(512, input_shape=(54,), activation='relu'),
    Dense(256, activation='relu'),
    Dense(256, activation='relu'),
    Dense(1)
])

rl = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]
path = 'C:\\프로그램\\ai5\\_save\\ml04\\'

for i, lr in enumerate(rl):
    model.compile(loss='mse', optimizer=Adam(learning_rate=lr), metrics=['mae'])
    
    # EarlyStopping과 ModelCheckpoint 설정
    es = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)
    
    date = datetime.datetime.now().strftime("%m%d_%H%M")
    filepath = f"{path}ml04_{i+1}_{date}_{{epoch:04d}}-{{val_loss:.4f}}.hdf5"
    mcp = ModelCheckpoint(filepath=filepath, monitor='val_loss', mode='auto', save_best_only=True)

    start = time.time()
    hist = model.fit(x_train_pca, y_train, epochs=200, batch_size=2048, validation_split=0.2, 
                     callbacks=[es, mcp], verbose=0)
    end = time.time()

    # 4. 평가 및 예측
    loss = model.evaluate(x_test, y_test, verbose=0)
    y_predict = model.predict(x_test, verbose=0)
    r2 = r2_score(y_test, y_predict)
    
    print("===============================================")
    print(f"Learning Rate: {lr}")
    print(f"테스트 손실 (Loss): {loss}")
    print(f"r2 스코어: {r2:.4f}")
    print(f"걸린 시간: {round(end - start, 2)} 초")
    print("===============================================")
'''
===============================================
Learning Rate: 0.1
테스트 손실 (Loss): [1.9483428001403809, 0.8112379908561707]
r2 스코어: -0.0000
걸린 시간: 11.66 초
===============================================
===============================================
Learning Rate: 0.01
테스트 손실 (Loss): [1.9483674764633179, 0.8103684782981873]
r2 스코어: -0.0000
걸린 시간: 9.26 초
===============================================
===============================================
Learning Rate: 0.005
테스트 손실 (Loss): [1.948351502418518, 0.8111173510551453]
r2 스코어: -0.0000
걸린 시간: 4.35 초
===============================================
===============================================
Learning Rate: 0.001
테스트 손실 (Loss): [1.9483345746994019, 0.8112974762916565]
r2 스코어: -0.0000
걸린 시간: 8.67 초
===============================================
Learning Rate: 0.0005
테스트 손실 (Loss): [1.9483140707015991, 0.8134095072746277]
r2 스코어: -0.0000
걸린 시간: 8.63 초
===============================================
===============================================
Learning Rate: 0.0001
테스트 손실 (Loss): [1.9483088254928589, 0.8141763806343079]
r2 스코어: -0.0000
걸린 시간: 4.44 초
===============================================
'''