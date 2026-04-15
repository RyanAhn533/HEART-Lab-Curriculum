import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1/255,

    horizontal_flip=True, #수평 뒤집기
    vertical_flip=True, #수직뒤집기
    width_shift_range=0.1, #평행이동
    height_shift_range=0.1, #평행이동 수직
    rotation_range=5, #정해진 각도만큼 이미지 회전
    zoom_range=1.2, #축소 또는 확대
    shear_range=0.7, # 좌표
    fill_mode="nearest", #비율에 맞춰서 채워라
    
    )

test_datagen = ImageDataGenerator(
    rescale=1./255,) #테스트 데이터는 절대 변환하지않는다. 데이터 조작.
path_train = 'C:/프로그램/ai5/_data/image/brain/train/'
path_test =   'C:/프로그램/ai5/_data/image/brain/test/'

xy_train = train_datagen.flow_from_directory(
    path_test, target_size=(200, 200), #(10,200,200,1)->16개가 나옴 (batchsize, targetsize, channel) 
    #found 160 - xy_train -> batchsize 10 -> 16개있음
    batch_size=10, 
#요 폴더에 있는걸 전부 수치화 해라
class_mode='binary',

color_mode='grayscale',
shuffle=True)
xy_test = test_datagen.flow_from_directory(
    path_test, target_size=(200, 200), #(10,200,200,1)->16개가 나옴 (batchsize, targetsize, channel) 
    #found 160 - xy_train -> batchsize 10 -> 16개있음
    batch_size=10, 
#요 폴더에 있는걸 전부 수치화 해라
class_mode='binary',

color_mode='grayscale',
shuffle=True)
#Found 160 images belonging to 2 classes. xy_train
#Found 120 images belonging to 2 classes. xy_test

# array([0., 1., 0., 0., 0., 1., 1., 1., 0., 1.]
print(xy_train[0]) #x만 뽑기
print(xy_train[0][0]) # 첫번째 배치의 Y

print(xy_train[0][1]) # 
#print(xy_train[0].shape) #AttributeError: 'DirectoryIterator' object has no attribute 'shape'
print(xy_train[0][0].shape)#(10, 200, 200, 1)
#print(xy_train[16])#ValueError: Asked to retrieve element 16, but the Sequence has length 12
#print(xy_train[15][2])#ValueError: Asked to retrieve element 16, but the Sequence has length 12
