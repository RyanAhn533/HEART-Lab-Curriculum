import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1/255
    )

test_datagen = ImageDataGenerator(
    rescale=1./255,) #테스트 데이터는 절대 변환하지않는다. 데이터 조작.
path_train = 'C:/프로그램/ai5/_data/image/rps/'

xy_train = train_datagen.flow_from_directory(
    path_train, target_size=(200, 200), #(10,200,200,1)->16개가 나옴 (batchsize, targetsize, channel) 
    #found 160 - xy_train -> batchsize 10 -> 16개있음
    batch_size=30, 
#요 폴더에 있는걸 전부 수치화 해라
#class_mode='categorical', #원핫인코딩 되어서 나옴
class_mode='sparse', #다중분류의 원핫 이전 모습
color_mode='rgb', # 
#color_mode='None' - 아무것도 안나옴
shuffle=True)
#y값 무시해버리기 위해서 다중분류 sparse로한다.
print(xy_train[0][0].shape)