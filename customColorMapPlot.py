import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

colors = ["blue", "green", "yellow"]
cmap = ListedColormap(colors)

data = np.random.randn(10, 10)

plt.imshow(data, cmap=cmap)
plt.colorbar()
plt.show()
