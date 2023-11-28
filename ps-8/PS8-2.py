#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import matplotlib.pyplot as plt


# In[19]:


ts = 0.0
te = 50.0
N = 10000
h = (te-ts)/N


# In[20]:


def f_x(x,y,z):
    return 10*(y - x)


def f_y(x,y,z):
    return 28*x -y-x*z


def f_z(x,y,z):
    return x*y-(8/3)*z


# In[23]:


def f(r):
    x = r[0]
    y = r[1]
    z = r[2]
    return np.array([f_x(x,y,z), f_y(x,y,z), f_z(x,y,z)], float)

tpoints = np.arange(ts, te, h)
xpoints = []
ypoints = []
zpoints = []
R = np.array([0,1,0], float)

for t in tpoints:
    xpoints.append(R[0])
    ypoints.append(R[1])
    zpoints.append(R[2])
    k1 = h * f(R)
    k2 = h * f(R+0.5*k1 + 0.5 * h)
    k3 = h * f(R+0.5*k2 + 0.5 * h)
    k4 = h * f(R+k3+ h)
    R += (k1 + 2 * k2 + 2 * k3 + k4)/6


# In[24]:


plt.plot(tpoints,ypoints)
plt.xlabel('t')
plt.ylabel('y')
plt.title('y_vs_t')
plt.savefig("y_vs_t.png")
plt.show()

plt.plot(xpoints,zpoints)
plt.xlabel('x')
plt.ylabel('z')
plt.title('z_vs_x')
plt.savefig("z_vs_x.png")
plt.show()


# In[ ]:




