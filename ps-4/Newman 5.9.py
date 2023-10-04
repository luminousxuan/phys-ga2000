#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy
from numpy import ones,copy,cos,tan,pi,linspace
import matplotlib.pyplot as plt


# In[2]:


rho = 6.022 * (10 ** 28)
Tdebye = 428
volume = 10 ** -3
boltzmann = 1.38*(10**-23)
coefficient =  9*volume*rho*boltzmann/(Tdebye**3)
#Root
def gaussxw(N):
    a = np.linspace(3,4*N-1,N)/(4*N+2)
    x = np.cos(pi*a+1/(8*N*N*tan(a)))
    
    epsilon = 1e-15
    delta = 1.0
    while delta>epsilon:
        p0 = np.ones(N,float)
        p1 = np.copy(x)
        for k in range(1,N):
            p0,p1 = p1,((2*k+1)*x*p1-k*p0)/(k+1)
        dp = (N+1)*(p0-x*p1)/(1-x*x)
        dx = p1/dp
        x -= dx
        delta = max(abs(dx))
    w = 2*(N+1)*(N+1)/(N*N*(1-x*x)*dp*dp)
    return x,w
def gaussxwab(N,a,b):
    x,w = gaussxw(N)
    return 0.5*(b-a)*x+0.5*(b+a),0.5*(b-a)*w


# In[5]:


def Cv(T,N):
    def f(x):
        return ((T**3)*(x**4*np.exp(x))/((np.exp(x)-1)**2))
    x, w = gaussxwab(N, 0, Tdebye/T)
    integral = 0.0
    for k in range(N):
        integral += w[k]*f(x[k])
    return integral


# In[6]:


HClist = []
for i in np.arange(5, 500, 1):
    newterm = Cv(i, 50) * coefficient
    HClist.append(newterm)


plt.plot(np.arange(5, 500, 1), HClist)
plt.xlabel("Temp (K)")
plt.ylabel("Specific Heat (J/K)")
plt.title("Heat Capacity")
plt.savefig("HeatCapacity.png")
plt.show()


# In[25]:


Convlist = []
for i in np.arange(10, 80, 10):
    newterm = Cv(100, i)* coefficient
    Convlist.append(newterm)

plt.plot(np.arange(10, 80, 10), Convlist)
plt.xlabel("Sample Size,N")
plt.ylabel("Specific Heat at T = 200K (J/K)")
plt.title("Convergence test of different sample size\n")
plt.savefig("Conv_200k")
plt.show()


# In[27]:


Convlist2 = []
for i in np.arange(10, 80, 10):
    newterm = Cv(50, i)* coefficient
    Convlist2.append(newterm)

plt.plot(np.arange(10, 80, 10), Convlist2)
plt.xlabel("Sample Size,N")
plt.ylabel("Specific Heat at T = 50K (J/K)")
plt.title("Convergence test of different sample size\n")
plt.savefig("Conv_50k")
plt.show()


# In[ ]:




