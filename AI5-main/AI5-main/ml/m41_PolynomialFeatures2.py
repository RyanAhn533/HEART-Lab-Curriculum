#데이터 증폭이다

import numpy as np
from sklearn.preprocessing import PolynomialFeatures

x = np.arange(12).reshape(4,3)
print(x)
# [[ 0  1  2]
#  [ 3  4  5]
#  [ 6  7  8]
#  [ 9 10 11]]

pf = PolynomialFeatures(degree=2, include_bias=True)
x_pf = pf.fit_transform(x)
print(x_pf)

# [[  1.   0.   1.   2.   0.   0.   0.   1.   2.   4.]
#  [  1.   3.   4.   5.   9.  12.  15.  16.  20.  25.]
#  [  1.   6.   7.   8.  36.  42.  48.  49.  56.  64.]
#  [  1.   9.  10.  11.  81.  90.  99. 100. 110. 121.]]



print("==========================================")
pf2 = PolynomialFeatures(degree=3, include_bias=True)
x_pf2 = pf2.fit_transform(x)
print(x_pf2)