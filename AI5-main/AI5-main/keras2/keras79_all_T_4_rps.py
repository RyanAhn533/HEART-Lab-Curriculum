<<<<<<< HEAD
<<<<<<< HEAD
from tensorflow.keras.applications import VGG16, ResNet50, DenseNet121, MobileNetV2, EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

# 랜덤 시드 고정
tf.random.set_seed(333)
np.random.seed(333)

# 모델 리스트
model_list = [
    VGG16(include_top=False, input_shape=(100, 100, 3)),
    ResNet50(include_top=False, input_shape=(100, 100, 3)),
    DenseNet121(include_top=False, input_shape=(100, 100, 3)),
    MobileNetV2(include_top=False, input_shape=(100, 100, 3)),
    EfficientNetB0(include_top=False, input_shape=(100, 100, 3))
]

# 데이터 로드
np_path = 'C:/ai5/_data/image/rps/'
x_train = np.load(np_path + "keras45_03_x_train.npy")
y_train = np.load(np_path + "keras45_03_y_train.npy")
x_test = np.load(np_path + "keras45_03_x_test.npy")
y_test = np.load(np_path + "keras45_03_y_test.npy")

# 데이터 정규화
x_train = x_train / 255.0
x_test = x_test / 255.0

# 데이터 분리
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# 데이터 크기 출력
print(f"x_train shape: {x_train.shape}, x_val shape: {x_val.shape}, x_test shape: {x_test.shape}")
print(f"y_train shape: {y_train.shape}, y_val shape: {y_val.shape}, y_test shape: {y_test.shape}")

# 모델별 학습 및 평가
for model_i in model_list:
    model_i.trainable = False  # 사전 학습 모델 고정

    # 모델 구성
    model = Sequential([
        model_i,
        GlobalAveragePooling2D(),
        Dense(100, activation='relu'),
        Dense(3, activation='softmax')  # 3개의 클래스
    ])

    # 모델 컴파일
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # EarlyStopping 설정
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

    # 학습
    hist = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=32,
        epochs=50,
        callbacks=[es],
        verbose=1
    )

    # 평가
    loss, acc = model.evaluate(x_test, y_test, verbose=1)
    print(f"모델명: {model_i.name}, Loss: {loss:.4f}, Accuracy: {acc:.2f}")

    # 예측 및 정확도 계산
    y_pred = np.argmax(model.predict(x_test), axis=1)
    print(f"정확도 (accuracy_score): {accuracy_score(y_test, y_pred):.2f}")
=======
from tensorflow.keras.applications import VGG16, ResNet50, DenseNet121, MobileNetV2, EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

# 랜덤 시드 고정
tf.random.set_seed(333)
np.random.seed(333)

# 모델 리스트
model_list = [
    VGG16(include_top=False, input_shape=(100, 100, 3)),
    ResNet50(include_top=False, input_shape=(100, 100, 3)),
    DenseNet121(include_top=False, input_shape=(100, 100, 3)),
    MobileNetV2(include_top=False, input_shape=(100, 100, 3)),
    EfficientNetB0(include_top=False, input_shape=(100, 100, 3))
]

# 데이터 로드
np_path = 'C:/ai5/_data/image/rps/'
x_train = np.load(np_path + "keras45_03_x_train.npy")
y_train = np.load(np_path + "keras45_03_y_train.npy")
x_test = np.load(np_path + "keras45_03_x_test.npy")
y_test = np.load(np_path + "keras45_03_y_test.npy")

# 데이터 정규화
x_train = x_train / 255.0
x_test = x_test / 255.0

# 데이터 분리
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# 데이터 크기 출력
print(f"x_train shape: {x_train.shape}, x_val shape: {x_val.shape}, x_test shape: {x_test.shape}")
print(f"y_train shape: {y_train.shape}, y_val shape: {y_val.shape}, y_test shape: {y_test.shape}")

# 모델별 학습 및 평가
for model_i in model_list:
    model_i.trainable = False  # 사전 학습 모델 고정

    # 모델 구성
    model = Sequential([
        model_i,
        GlobalAveragePooling2D(),
        Dense(100, activation='relu'),
        Dense(3, activation='softmax')  # 3개의 클래스
    ])

    # 모델 컴파일
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # EarlyStopping 설정
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

    # 학습
    hist = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=32,
        epochs=50,
        callbacks=[es],
        verbose=1
    )

    # 평가
    loss, acc = model.evaluate(x_test, y_test, verbose=1)
    print(f"모델명: {model_i.name}, Loss: {loss:.4f}, Accuracy: {acc:.2f}")

    # 예측 및 정확도 계산
    y_pred = np.argmax(model.predict(x_test), axis=1)
    print(f"정확도 (accuracy_score): {accuracy_score(y_test, y_pred):.2f}")
>>>>>>> cd855f8 (message)
=======
from tensorflow.keras.applications import VGG16, ResNet50, DenseNet121, MobileNetV2, EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

# 랜덤 시드 고정
tf.random.set_seed(333)
np.random.seed(333)

# 모델 리스트
model_list = [
    VGG16(include_top=False, input_shape=(100, 100, 3)),
    ResNet50(include_top=False, input_shape=(100, 100, 3)),
    DenseNet121(include_top=False, input_shape=(100, 100, 3)),
    MobileNetV2(include_top=False, input_shape=(100, 100, 3)),
    EfficientNetB0(include_top=False, input_shape=(100, 100, 3))
]

# 데이터 로드
np_path = 'C:/ai5/_data/image/rps/'
x_train = np.load(np_path + "keras45_03_x_train.npy")
y_train = np.load(np_path + "keras45_03_y_train.npy")
x_test = np.load(np_path + "keras45_03_x_test.npy")
y_test = np.load(np_path + "keras45_03_y_test.npy")

# 데이터 정규화
x_train = x_train / 255.0
x_test = x_test / 255.0

# 데이터 분리
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# 데이터 크기 출력
print(f"x_train shape: {x_train.shape}, x_val shape: {x_val.shape}, x_test shape: {x_test.shape}")
print(f"y_train shape: {y_train.shape}, y_val shape: {y_val.shape}, y_test shape: {y_test.shape}")

# 모델별 학습 및 평가
for model_i in model_list:
    model_i.trainable = False  # 사전 학습 모델 고정

    # 모델 구성
    model = Sequential([
        model_i,
        GlobalAveragePooling2D(),
        Dense(100, activation='relu'),
        Dense(3, activation='softmax')  # 3개의 클래스
    ])

    # 모델 컴파일
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # EarlyStopping 설정
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

    # 학습
    hist = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=32,
        epochs=50,
        callbacks=[es],
        verbose=1
    )

    # 평가
    loss, acc = model.evaluate(x_test, y_test, verbose=1)
    print(f"모델명: {model_i.name}, Loss: {loss:.4f}, Accuracy: {acc:.2f}")

    # 예측 및 정확도 계산
    y_pred = np.argmax(model.predict(x_test), axis=1)
    print(f"정확도 (accuracy_score): {accuracy_score(y_test, y_pred):.2f}")
>>>>>>> 70eabacb3fa0ad4089229f1c83ce2c346b0e48a8
    