<<<<<<< HEAD
<<<<<<< HEAD
# 다른말로 swish = SiLu

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

def leaky_rely(x, alpha=0.01):
    return np.where(x>0,x, alpha * x )

y = leaky_rely(x)

plt.plot(x,y)
plt.grid()
=======
# 다른말로 swish = SiLu

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

def leaky_rely(x, alpha=0.01):
    return np.where(x>0,x, alpha * x )

y = leaky_rely(x)

plt.plot(x,y)
plt.grid()
>>>>>>> cd855f8 (message)
=======
# 다른말로 swish = SiLu

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

def leaky_rely(x, alpha=0.01):
    return np.where(x>0,x, alpha * x )

y = leaky_rely(x)

plt.plot(x,y)
plt.grid()
>>>>>>> 70eabacb3fa0ad4089229f1c83ce2c346b0e48a8
plt.show()