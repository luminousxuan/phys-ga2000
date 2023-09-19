#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[14]:


# Mandelbrot function
def mandelbrot(x, y, iteration):
    c = complex(x, y)
    z = 0
    for i in range(1, iteration):
        if abs(z) > 2:
            return False
        z = z ** 2 + c
    return True


# In[16]:


N=1000
iteration = 100
X = np.linspace(-2,2,N)
Y = np.linspace(-2,2,N)
Mandelbrot = np.zeros((N,N))
index1 =-1
for x in X:
    index1 += 1
    index2 =-1
    for y in Y:
        index2 +=1
        boolean = mandelbrot(x,y,iteration)
        if boolean:
            Mandelbrot[index1,index2] = index1
plt.imshow(Mandelbrot,extent=(-2, 2, 2, -2))
plt.xlabel('Imaginary')
plt.ylabel('Real')
plt.title('Mandelbrot Set')
plt.jet()
plt.show()


# In[ ]:




