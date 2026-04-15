from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train/255.
x_test = x_test/255.

train_datagen =  ImageDataGenerator(
    #rescale=1./255,              # 이미지를 수치화 할 때 0~1 사이의 값으로 (스케일링 한 데이터로 사용)
    horizontal_flip=True,        # 수평 뒤집기   <- 데이터 증폭 
    # vertical_flip=True,          # 수직 뒤집기 (상하좌우반전) <- 데이터 증폭
    width_shift_range=0.2,       # 평행이동  <- 데이터 증폭
    # height_shift_range=0.1,      # 평행이동 수직  <- 데이터 증폭
    rotation_range=15,            # 각도 조절 (정해진 각도만큼 이미지 회전)
    # zoom_range=1.2,              # 축소 또는 확대
    # shear_range=0.7,             # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환 (찌부시키기)
    fill_mode='nearest',         # 10% 이동 시 한쪽은 소실, 한쪽은 가까이에 있던 부분의 이미지로 채워짐
)

augment_size = 40000 # 똑같은 타일을 40000개 찍었다.
print(x_train.shape[0])
 
randidx = np.random.randint(x_train.shape[0], size = augment_size) #60000, size=40000
#randint -> 새로 생성할 데이터 4만개 생성댐
print(randidx)
print(np.min(randidx), np.max(randidx)) # 0 59997
x_augmented = x_train[randidx].copy()  #.copy 하면 메모리 안전빵!
y_augmented = y_train[randidx].copy()

x_augmented = x_augmented.reshape(
                                  x_augmented.shape[0], # 40000
                                  x_augmented.shape[1], #28
                                  x_augmented.shape[2], 1)#28
print(x_augmented.shape) # (40000 28 28 1)


print(x_augmented[0].shape) # (28 28 1)



print(x_augmented.shape, y_augmented.shape) #(40000, 28, 28) (40000,)
x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False,
).next()[0]
#왜 셔플하면 안되는가?
print(x_augmented[0].shape) #(28, 28, 1)
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)
print(x_train.shape, x_test.shape)


x_train = np.concatenate((x_train,x_augmented), axis = 0)
print(x_train.shape)
y_train = np.concatenate((y_train, y_augmented), axis = 0)
print(y_train.shape)

###############맹그러봐############