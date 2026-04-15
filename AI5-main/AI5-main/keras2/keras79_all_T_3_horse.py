<<<<<<< HEAD
<<<<<<< HEAD
from tensorflow.keras.applications import VGG16, ResNet50, DenseNet121, MobileNetV2, EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import tensorflow as tf
from sklearn.metrics import accuracy_score
import time

tf.random.set_seed(333)
np.random.seed(333)

# 사전 학습 모델 리스트
model_list = [
    VGG16(include_top=False, input_shape=(100, 100, 3)),
    ResNet50(include_top=False, input_shape=(100, 100, 3)),
    DenseNet121(include_top=False, input_shape=(100, 100, 3)),
    MobileNetV2(include_top=False, input_shape=(100, 100, 3)),
    EfficientNetB0(include_top=False, input_shape=(100, 100, 3))
]

# 데이터 로드 및 전처리
path_train = 'C:/ai5/_data/image/horse_human/'
train_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory(
    path_train, target_size=(100, 100), 
    batch_size=30000, 
    class_mode='binary',  # 이진 분류
    color_mode='rgb',
    shuffle=True
)

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(
    xy_train[0][0], xy_train[0][1], train_size=0.7, random_state=3
)

# 모델 학습 및 평가
for model_i in model_list:
    model_i.trainable = False

    # 모델 구성
    model = Sequential()
    model.add(model_i)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # 이진 분류용 출력층

    # 모델 컴파일
    model.compile(
        loss='binary_crossentropy',  # 이진 분류 손실 함수
        optimizer='adam',
        metrics=['accuracy']
    )

    # 모델 학습
    start = time.time()
    model.fit(
        x_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1
    )
    end = time.time()

    # 평가
    loss, acc = model.evaluate(x_test, y_test, verbose=1)
    print(f"모델명: {model_i.name}, Loss: {loss:.4f}, Accuracy: {acc:.2f}, 학습 시간: {end - start:.2f}초")

    # 예측 및 정확도 평가
    y_pred = (model.predict(x_test) > 0.5).astype("int32")
    print(f"정확도 (accuracy_score): {accuracy_score(y_test, y_pred):.2f}")

'''
모델명: vgg16, Loss: 0.0181, Accuracy: 0.99, 학습 시간: 18.99초
정확도 (accuracy_score): 0.99

모델명: resnet50, Loss: 0.3230, Accuracy: 0.85, 학습 시간: 19.69초
정확도 (accuracy_score): 0.85

모델명: densenet121, Loss: 0.0007, Accuracy: 1.00, 학습 시간: 25.09초
정확도 (accuracy_score): 1.00

모델명: mobilenetv2_1.00_224, Loss: 0.0085, Accuracy: 1.00, 학습 시간: 12.36초
정확도 (accuracy_score): 1.00

모델명: efficientnetb0, Loss: 0.6899, Accuracy: 0.56, 학습 시간: 20.77초
정확도 (accuracy_score): 0.56
=======
from tensorflow.keras.applications import VGG16, ResNet50, DenseNet121, MobileNetV2, EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import tensorflow as tf
from sklearn.metrics import accuracy_score
import time

tf.random.set_seed(333)
np.random.seed(333)

# 사전 학습 모델 리스트
model_list = [
    VGG16(include_top=False, input_shape=(100, 100, 3)),
    ResNet50(include_top=False, input_shape=(100, 100, 3)),
    DenseNet121(include_top=False, input_shape=(100, 100, 3)),
    MobileNetV2(include_top=False, input_shape=(100, 100, 3)),
    EfficientNetB0(include_top=False, input_shape=(100, 100, 3))
]

# 데이터 로드 및 전처리
path_train = 'C:/ai5/_data/image/horse_human/'
train_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory(
    path_train, target_size=(100, 100), 
    batch_size=30000, 
    class_mode='binary',  # 이진 분류
    color_mode='rgb',
    shuffle=True
)

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(
    xy_train[0][0], xy_train[0][1], train_size=0.7, random_state=3
)

# 모델 학습 및 평가
for model_i in model_list:
    model_i.trainable = False

    # 모델 구성
    model = Sequential()
    model.add(model_i)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # 이진 분류용 출력층

    # 모델 컴파일
    model.compile(
        loss='binary_crossentropy',  # 이진 분류 손실 함수
        optimizer='adam',
        metrics=['accuracy']
    )

    # 모델 학습
    start = time.time()
    model.fit(
        x_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1
    )
    end = time.time()

    # 평가
    loss, acc = model.evaluate(x_test, y_test, verbose=1)
    print(f"모델명: {model_i.name}, Loss: {loss:.4f}, Accuracy: {acc:.2f}, 학습 시간: {end - start:.2f}초")

    # 예측 및 정확도 평가
    y_pred = (model.predict(x_test) > 0.5).astype("int32")
    print(f"정확도 (accuracy_score): {accuracy_score(y_test, y_pred):.2f}")

'''
모델명: vgg16, Loss: 0.0181, Accuracy: 0.99, 학습 시간: 18.99초
정확도 (accuracy_score): 0.99

모델명: resnet50, Loss: 0.3230, Accuracy: 0.85, 학습 시간: 19.69초
정확도 (accuracy_score): 0.85

모델명: densenet121, Loss: 0.0007, Accuracy: 1.00, 학습 시간: 25.09초
정확도 (accuracy_score): 1.00

모델명: mobilenetv2_1.00_224, Loss: 0.0085, Accuracy: 1.00, 학습 시간: 12.36초
정확도 (accuracy_score): 1.00

모델명: efficientnetb0, Loss: 0.6899, Accuracy: 0.56, 학습 시간: 20.77초
정확도 (accuracy_score): 0.56
>>>>>>> cd855f8 (message)
=======
from tensorflow.keras.applications import VGG16, ResNet50, DenseNet121, MobileNetV2, EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import tensorflow as tf
from sklearn.metrics import accuracy_score
import time

tf.random.set_seed(333)
np.random.seed(333)

# 사전 학습 모델 리스트
model_list = [
    VGG16(include_top=False, input_shape=(100, 100, 3)),
    ResNet50(include_top=False, input_shape=(100, 100, 3)),
    DenseNet121(include_top=False, input_shape=(100, 100, 3)),
    MobileNetV2(include_top=False, input_shape=(100, 100, 3)),
    EfficientNetB0(include_top=False, input_shape=(100, 100, 3))
]

# 데이터 로드 및 전처리
path_train = 'C:/ai5/_data/image/horse_human/'
train_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory(
    path_train, target_size=(100, 100), 
    batch_size=30000, 
    class_mode='binary',  # 이진 분류
    color_mode='rgb',
    shuffle=True
)

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(
    xy_train[0][0], xy_train[0][1], train_size=0.7, random_state=3
)

# 모델 학습 및 평가
for model_i in model_list:
    model_i.trainable = False

    # 모델 구성
    model = Sequential()
    model.add(model_i)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # 이진 분류용 출력층

    # 모델 컴파일
    model.compile(
        loss='binary_crossentropy',  # 이진 분류 손실 함수
        optimizer='adam',
        metrics=['accuracy']
    )

    # 모델 학습
    start = time.time()
    model.fit(
        x_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1
    )
    end = time.time()

    # 평가
    loss, acc = model.evaluate(x_test, y_test, verbose=1)
    print(f"모델명: {model_i.name}, Loss: {loss:.4f}, Accuracy: {acc:.2f}, 학습 시간: {end - start:.2f}초")

    # 예측 및 정확도 평가
    y_pred = (model.predict(x_test) > 0.5).astype("int32")
    print(f"정확도 (accuracy_score): {accuracy_score(y_test, y_pred):.2f}")

'''
모델명: vgg16, Loss: 0.0181, Accuracy: 0.99, 학습 시간: 18.99초
정확도 (accuracy_score): 0.99

모델명: resnet50, Loss: 0.3230, Accuracy: 0.85, 학습 시간: 19.69초
정확도 (accuracy_score): 0.85

모델명: densenet121, Loss: 0.0007, Accuracy: 1.00, 학습 시간: 25.09초
정확도 (accuracy_score): 1.00

모델명: mobilenetv2_1.00_224, Loss: 0.0085, Accuracy: 1.00, 학습 시간: 12.36초
정확도 (accuracy_score): 1.00

모델명: efficientnetb0, Loss: 0.6899, Accuracy: 0.56, 학습 시간: 20.77초
정확도 (accuracy_score): 0.56
>>>>>>> 70eabacb3fa0ad4089229f1c83ce2c346b0e48a8
'''