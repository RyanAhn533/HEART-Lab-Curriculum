import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, BatchNormalization
import time
from sklearn.model_selection import train_test_split

#1. data (시간체크)
start_time=time.time()
train_datagen = ImageDataGenerator(
    rescale=1./255,         # 이미지 스케일링
    horizontal_flip= True,  # 수평 뒤집기
    vertical_flip=True,     # 수집 뒤집기
    width_shift_range=0.1,  # 평행이동
    height_shift_range=0.1, # 평행이동 수직
    rotation_range=1,       # 각도 조절
    zoom_range=0.2,         # 축소 또는 확대
    shear_range=0.7,        # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환    
    fill_mode='nearest',    # 이미지가 이동할 때 가장 가까운 곳의 색을 채운다는 뜻
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
)

path = 'C:\\프로그램\\ai5\\_data\\kaggle\\Biggest_gender\\faces\\man\\'
# path_test = './_data/image/cat_and_dog/Test/'


xy_train1 = train_datagen.flow_from_directory(
    path, 
    target_size=(100,100),
    batch_size=30000,
    class_mode='binary',
    color_mode='rgb', 
    shuffle=True, # False로 할 경우 print(xy_train) 값이 array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)) -> 다 0으로 됨
)  # Found 160 images belonging to 2 classes.

xy_train2 = test_datagen.flow_from_directory(
    path,            
    target_size=(100,100),  
    batch_size=25000,          
    class_mode='binary',  
    color_mode='rgb',  
    shuffle=True, 
)
# xy_test = test_datagen.flow_from_directory(
#     path_test, 
#     target_size=(100,100),
#     batch_size=20000,
#     class_mode='binary',
#     color_mode='rgb',
#     shuffle=False, # 어지간하면 셔플할 필요 없음.
# )  # Found 120 images belonging to 2 classes.



np_path = 'c:/프로그램/ai5/_data/kaggle/biggest_gender/'
np.save(np_path + 'man_x_train1.npy', arr=xy_train1[0][0])
np.save(np_path + 'man_y_train1.npy', arr=xy_train1[0][1])
np.save(np_path + 'man_x_train2.npy', arr=xy_train2[0][0])
np.save(np_path + 'man_y_train2.npy', arr=xy_train2[0][1])
end_time=time.time()
print("데이터 처리 걸린 시간 :", round(end_time-start_time,2),'초')