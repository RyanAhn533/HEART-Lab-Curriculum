import pandas as pd
import numpy as np


df = pd.DataFrame({'A' : [1,2,3,4,5],
                   'B' : [10, 20, 30, 40, 50],
                   'C' : [5, 4, 3, 2, 1],
                   'D' : [3, 7, 5, 1, 4]})

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
print(sns.__version__)
print(matplotlib.__version__)
#sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(),
            square=True,
            annot=True,
            cbar=True)
plt.show()