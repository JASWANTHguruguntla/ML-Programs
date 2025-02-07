import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
x = np.random.rand(50)
y = np.random.rand(50)
colors = np.random.rand(50)
cmap = ListedColormap(['red', 'green', 'blue'])
plt.scatter(x, y, c=colors, cmap=cmap)
plt.colorbar() 
plt.show()
