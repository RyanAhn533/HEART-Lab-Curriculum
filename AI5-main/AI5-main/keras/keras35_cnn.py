from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D
#Convolution 2D = Conv2d

model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(5, 5, 1))) # (4, 4, 10) 
#(2,2) = 네모 쪼개기 
# (5, 5, 1) 가로 세로 칼라 V
#흑백사진 복원은? -> RGB는 세장에서 하나 빼면 되면, 흑백에서 칼라는 ? 임의지정?
#5 5 1 -> 5x5 1장 - 흑백이니 한장 - (2,2) 둘 다 마음대로 설정 가능 하이퍼 파라미터 튜닝
#5 5 1 을 2 2 로 나눠주면 4 x 4 개가 나옴 -> 반복하다보면 겹치기때문에 큰놈은 크게 작은놈 더 작게 바뀜
# convolutuion 계속하면 ? 압축? -> 큰놈은 더 커지고 작은놈은 더 작아지고 /4 x 4 짜리 / 레이어를 통과시키면 10개가 된다
# convolution 을 너무 많이하면 데이터가 소실된다 -> 특성이 작아지기 때문에 = 겹치는 게 작아진다
#500 400 사진
# cornersize?
model.add(Conv2D(5, (2,2))) (3, 3, 5)

model.summary()