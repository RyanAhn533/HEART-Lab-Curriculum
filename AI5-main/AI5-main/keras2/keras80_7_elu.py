<<<<<<< HEAD
<<<<<<< HEAD
# 다른말로 swish = SiLu

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

def elu(x):
    return np.where(x > 0, x, 1 * (np.exp(x) - 1))

y = elu(x)

plt.plot(x,y)
plt.grid()
=======
# 다른말로 swish = SiLu

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

def elu(x):
    return np.where(x > 0, x, 1 * (np.exp(x) - 1))

y = elu(x)

plt.plot(x,y)
plt.grid()
>>>>>>> cd855f8 (message)
=======
# 다른말로 swish = SiLu

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

def elu(x):
    return np.where(x > 0, x, 1 * (np.exp(x) - 1))

y = elu(x)

plt.plot(x,y)
plt.grid()
>>>>>>> 70eabacb3fa0ad4089229f1c83ce2c346b0e48a8
plt.show()