import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
plt.rcParams['font.family'] = 'Malgun Gothic'

# 1. 데이터 생성
np.random.seed(777)
x = 2 * np.random.rand(100, 1) - 1  # -1부터 1까지 난수 생성

print(np.max(x), np.min(x))
y = 3 * x**2 + 2 * x + np.random.randn(100, 1)  # y = 3x^2 + 2x + (잡음 추가)

# rand : 0~1 사이의 균일분포 난수
# randn : 평균 0, 표준편차 1의 정규분포 난수

# 다항식 특성 생성
pf = PolynomialFeatures(degree=2, include_bias=False)
x_poly = pf.fit_transform(x)
print(x_poly)

# 2. 모델 정의
model = LinearRegression()      # 단순 선형 회귀
model2 = LinearRegression()     # 다항식 회귀

# 3. 훈련
model.fit(x, y)
model2.fit(x_poly, y)

# 4. 원본 데이터 시각화
plt.scatter(x, y, color='blue', label='Original Data')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Regression 예제')

# 5. 다항식 회귀 그래프 그리기
x_test = np.linspace(-1, 1, 100).reshape(-1, 1)  # 그래프 그릴 x 값 범위 설정
x_test_poly = pf.transform(x_test)              # 다항식 변환
y_plot = model.predict(x_test)                  # 단순 선형 회귀 예측
y_plot2 = model2.predict(x_test_poly)           # 다항식 회귀 예측

plt.plot(x_test,y_plot,  color='red', label='Linear Regression')
plt.plot(x_test, y_plot2, color='green', label='Polynomial Regression')

plt.legend()
plt.show()

