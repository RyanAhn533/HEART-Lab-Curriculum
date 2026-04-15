import tensorflow as tf
print(tf.__version__)
#physical_devisces 물리적인 경로
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)

if(gpus):
    print("쥐피유 돈다!!!")
else:
    print("쥐피유 없다!")