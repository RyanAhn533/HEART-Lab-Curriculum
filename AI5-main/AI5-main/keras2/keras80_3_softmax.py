<<<<<<< HEAD
<<<<<<< HEAD
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1, 5)

# def softmax(x):
#     return np.exp(x) / np.sum(np.exp(x))

softmax = lambda x : np.exp(x) / np.sum(np.exp(x))

y = softmax(x)

ratio = y
labels = y
plt.pie(ratio, labels, shadow=True, startangle=90)
=======
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1, 5)

# def softmax(x):
#     return np.exp(x) / np.sum(np.exp(x))

softmax = lambda x : np.exp(x) / np.sum(np.exp(x))

y = softmax(x)

ratio = y
labels = y
plt.pie(ratio, labels, shadow=True, startangle=90)
>>>>>>> cd855f8 (message)
=======
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1, 5)

# def softmax(x):
#     return np.exp(x) / np.sum(np.exp(x))

softmax = lambda x : np.exp(x) / np.sum(np.exp(x))

y = softmax(x)

ratio = y
labels = y
plt.pie(ratio, labels, shadow=True, startangle=90)
>>>>>>> 70eabacb3fa0ad4089229f1c83ce2c346b0e48a8
plt.show()