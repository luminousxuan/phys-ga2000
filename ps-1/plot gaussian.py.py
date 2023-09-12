import numpy as np
import matplotlib.pyplot as plt

mean = 0
sigma = 3
x = np.arange(-10,10,0.1)
y = np.multiply(np.power(np.sqrt(2*np.pi)*sigma,-1),np.exp(-np.power(x-mean,2)/2*sigma**2))

plt.plot(x, y,linewidth=1)
plt.title('Gaussian Distribution')
plt.show()