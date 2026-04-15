import autokeras as ak
import tensorflow as tf
import time
print(ak.__version__)
print(ak.__version__)

#1.데이터
(x_train,y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)

#2. 모델
model = ak.ImageClassifier(
    overwrite=False,
    max_trials=1,)

#3. 컴파일, 훈련
start_time = time.time()
model.fit(x_train, y_train, epochs=1, validation_split=0.15)
end_time = time.time()

##########최적의 출력 모델 ##################
best_model = model.export_model()
print(best_model.summary())

######## 최적의 모델 저장 ########
path = '.\\_save\\autokeras\\'
best_model.save(path + 'keras70_autokeras1.h5')

#4. 평가 예측
y_predict = model.predict(x_test)
results = best_model.evaluate(x_test, y_test)
print('moedl 결과 : ', results)

y_predict2 = best_model.predict(x_test)
results2 = best_model.evaluate(x_test, y_test)
print('moedl 결과 : ', results2)

print('걸린 시간은? : ', round(end_time - start_time, 2), '초')