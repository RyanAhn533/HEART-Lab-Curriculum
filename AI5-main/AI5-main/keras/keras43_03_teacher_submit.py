import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Input, MaxPooling2D
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization, MaxPool2D

start = time.time()

np_path = 'C:\\프로그램\\ai5\\_data\\_save\\save_npy\\'

x_train = np.load(np_path + "keras43_01_x_train1.npy")
y_train = np.load(np_path + "keras43_01_y_train1.npy")
x_test = np.load(np_path + "keras43_01_x_test.npy")
y_test = np.load(np_path + "keras43_01_y_test.npy")
print(x_train)
print(x_train.shape)

# 모델 로드
model_path = 'c:/프로그램/ai5/_save/keras42/_kaggle_cats_dog/k42_01_0802_1844_0019-0.3302.hdf5'
model = load_model(model_path)

# 모델 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



# CSV 파일 만들기
sampleSubmission = pd.read_csv('C:/프로그램/ai5/_data/kaggle/dogs_vs_cats/sample_submission.csv', index_col=0)
#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', round(loss[1],5))

y_pre = np.round(model.predict(x_test, batch_size=16))

end_time=time.time()


print("걸린 시간 :", round(end_time-start,2),'초')

### csv 파일 만들기 ###
y_submit = model.predict(x_test, batch_size=16)
y_submit = np.clip(y_submit, 1e-6, 1-(1e-6))
# 또는 sampleSubmission 길이에 맞추기 (필요시)
y_submit = y_submit[:len(sampleSubmission)]
sampleSubmission['label'] = y_submit
sampleSubmission.to_csv('C:/프로그램/ai5/_data/kaggle/dogs_vs_cats/' + "teacher0805_2.csv")