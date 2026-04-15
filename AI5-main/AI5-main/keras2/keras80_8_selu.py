<<<<<<< HEAD
<<<<<<< HEAD
# 다른말로 swish = SiLu

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

def selu(x, lambda_x=1.0507, alpha = 1.67326):
    return np.where(x>0, lambda_x*x, lambda_x*alpha*(np.exp(x)-1))

y = selu(x)

plt.plot(x, y)
plt.grid()
=======
# 다른말로 swish = SiLu

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

def selu(x, lambda_x=1.0507, alpha = 1.67326):
    return np.where(x>0, lambda_x*x, lambda_x*alpha*(np.exp(x)-1))

y = selu(x)

plt.plot(x, y)
plt.grid()
>>>>>>> cd855f8 (message)
=======
# 다른말로 swish = SiLu

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

def selu(x, lambda_x=1.0507, alpha = 1.67326):
    return np.where(x>0, lambda_x*x, lambda_x*alpha*(np.exp(x)-1))

y = selu(x)

plt.plot(x, y)
plt.grid()
>>>>>>> 70eabacb3fa0ad4089229f1c83ce2c346b0e48a8
plt.show()