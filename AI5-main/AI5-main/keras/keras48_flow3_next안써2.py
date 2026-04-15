from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen =  ImageDataGenerator(
    rescale=1./255,              # 이미지를 수치화 할 때 0~1 사이의 값으로 (스케일링 한 데이터로 사용)
    horizontal_flip=True,        # 수평 뒤집기   <- 데이터 증폭 
    # vertical_flip=True,          # 수직 뒤집기 (상하좌우반전) <- 데이터 증폭
    width_shift_range=0.2,       # 평행이동  <- 데이터 증폭
    # height_shift_range=0.1,      # 평행이동 수직  <- 데이터 증폭
    rotation_range=15,            # 각도 조절 (정해진 각도만큼 이미지 회전)
    # zoom_range=1.2,              # 축소 또는 확대
    # shear_range=0.7,             # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환 (찌부시키기)
    fill_mode='nearest',         # 10% 이동 시 한쪽은 소실, 한쪽은 가까이에 있던 부분의 이미지로 채워짐
)

augment_size = 100 # 똑같은 타일을 100개 찍었다.
 
xy_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1, 28, 28, 1),
    np.zeros(augment_size),
    batch_size=32,
    shuffle=False,
)
#print(xy_data)
print(type(xy_data)) # .next()가 있으면 <class 'tuple'>
# print(len(xy_data))
# print(xy_data[0][0].shape)
# print(xy_data.shape) #이건안댐
# print(len(xy_data)) #튜플에는 쉐이프가 없어
print(xy_data[1][0].shape)#(32, 28, 28, 1) 
print(xy_data[2][0].shape) #(32, 28, 28, 1)
print(xy_data[3][0].shape) #(4, 28, 28, 1)
print(xy_data[0][1].shape)#(32, 28, 28, 1) 



"""
plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7,7,i+1)
    plt.imshow(xy_data[0][i], cmap='gray')
    
plt.show()
"""