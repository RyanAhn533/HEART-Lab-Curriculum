import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
plt.rcParams['font.family'] = 'Malgun Gothic'

#1. 데이터
np.random.seed(777)
x = 2* np.random.rand(100,1) -1  # -1부터 1까지 난수생성

print(np.max(x), np.min(x))
y = 3*x**2 +2*x+np.random.randn(100,1)  # y=3x^2 + 2x +1 +

#rand : 0~1 사이의 균일분포 난수
#randn : 평균 0, 표편1 정규분포 난수

pf = PolynomialFeatures(degree=2, include_bias=False)
x_poly = pf.fit_transform(x)
print(x_poly)

#2. 모델
model = LinearRegression()
model2 = LinearRegression()

#3. 훈련
model.fit(x, y)
model2.fit(x_poly, y)

#원본 데이터 그리자.
plt.scatter(x, y, color='blue', label='Original Data')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Regression 예제')
#plt.show()

#다항식 회귀 그래프 그리기
x_plot = np.linspace(-1, 1, 100).reshape(-1,1)
x_plot_poly = pf.transform(x_plot)
y_plot = model.predict(x_plot)
y_plot2 = model.predict(x_plot_poly)
plt.plot(x_plot, y_plot, color='red', label='Polynomail Regression')
plt.plot(x_plot, y_plot, color='blue', label='기냥')

plt.legend()
plt.show()