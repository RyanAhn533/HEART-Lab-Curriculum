<<<<<<< HEAD
<<<<<<< HEAD
import numpy as np
import matplotlib.pyplot as plt

#def sigmoid(x):
#    return 1 / (1 + np.exp(-x))

sigmoid = lambda x : 1 / (1 + np.exp(-x))

x = np.arange(-5, 5, 0.1)
print(x)
print(len(x))

y = sigmoid(x)
plt.plot(x, y)
plt.grid()
=======
import numpy as np
import matplotlib.pyplot as plt

#def sigmoid(x):
#    return 1 / (1 + np.exp(-x))

sigmoid = lambda x : 1 / (1 + np.exp(-x))

x = np.arange(-5, 5, 0.1)
print(x)
print(len(x))

y = sigmoid(x)
plt.plot(x, y)
plt.grid()
>>>>>>> cd855f8 (message)
=======
import numpy as np
import matplotlib.pyplot as plt

#def sigmoid(x):
#    return 1 / (1 + np.exp(-x))

sigmoid = lambda x : 1 / (1 + np.exp(-x))

x = np.arange(-5, 5, 0.1)
print(x)
print(len(x))

y = sigmoid(x)
plt.plot(x, y)
plt.grid()
>>>>>>> 70eabacb3fa0ad4089229f1c83ce2c346b0e48a8
plt.show()