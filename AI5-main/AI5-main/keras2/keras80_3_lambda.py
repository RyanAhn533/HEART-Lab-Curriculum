<<<<<<< HEAD
<<<<<<< HEAD
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

#def relu(x):
#    return np.maximum(0, x)

relu = lambda x : np.maximum(0, x)

y = relu(x)

plt.plot(x, y)
plt.grid()
=======
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

#def relu(x):
#    return np.maximum(0, x)

relu = lambda x : np.maximum(0, x)

y = relu(x)

plt.plot(x, y)
plt.grid()
>>>>>>> cd855f8 (message)
=======
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

#def relu(x):
#    return np.maximum(0, x)

relu = lambda x : np.maximum(0, x)

y = relu(x)

plt.plot(x, y)
plt.grid()
>>>>>>> 70eabacb3fa0ad4089229f1c83ce2c346b0e48a8
plt.show()