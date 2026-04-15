#데이터 증폭이다

import numpy as np
from sklearn.preprocessing import PolynomialFeatures

x = np.arange(8).reshape(4,2)

print(x)
# [[0 1] 
#  [2 3] 
#  [4 5] 
#  [6 7]]

pf = PolynomialFeatures(degree=2, include_bias=True)
x_pf = pf.fit_transform(x)
print(x_pf)
#FALSE  y = x1 + x2 + x1^2 + x1x2 + x2^2 + b(default=1)
#   x1   x2  
# [[ 0.  1.  0.  0.  1.]
#  [ 2.  3.  4.  6.  9.]
#  [ 4.  5. 16. 20. 25.]
#  [ 6.  7. 36. 42. 49.]]

# Linear Regressor할 경우에는 성능이 좋음 / Tree 계열같은 건 bias의미 없음
#TRUE (b = 1로 넣어버림)
# [[ 1.  0.  1.  0.  0.  1.]
#  [ 1.  2.  3.  4.  6.  9.]
#  [ 1.  4.  5. 16. 20. 25.]
#  [ 1.  6.  7. 36. 42. 49.]]

##통상적으로
# 선형모델 (lr등)에 쓸경우에는 Include_bias = True를 써서 1만 있는 컬럼을 만드는게 좋고
# 왜나하면 y = wx+b의 바이어스=1의 역할을 하기 때문
# 비선형 모델 (rf, xgb등)에 쓸 경우에는 iclude_bias+False가 좋다

pf2 = PolynomialFeatures(degree=3, include_bias=False)
x_pf2 = pf2.fit_transform(x)
print(x_pf2)
# [[  0.   1.   0.   0.   1.   0.   0.   0.   1.]
#  [  2.   3.   4.   6.   9.   8.  12.  18.  27.]
#  [  4.   5.  16.  20.  25.  64.  80. 100. 125.]
#  [  6.   7.  36.  42.  49. 216. 252. 294. 343.]]