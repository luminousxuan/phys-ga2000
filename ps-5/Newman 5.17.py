#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from numpy import ones,copy,cos,tan,pi,linspace

def gaussxw(N):

    a = linspace(3,4*N-1,N)/(4*N+2)
    x = cos(pi*a+1/(8*N*N*tan(a)))

    epsilon = 1e-15
    delta = 1.0
    while delta>epsilon:
        p0 = ones(N,float)
        p1 = copy(x)
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


# In[2]:


def gamma(x, a):
    return x**(a-1)*np.exp(-x)


# In[3]:


def gauss_quad(p,q,N,a):
    x, w = gaussxwab(N,p,q)
    G = 0
    for k in range(N):
        G += w[k]*gamma(x[k], a)
    return G


# In[4]:


xrange = np.linspace(0, 5, 1000)
N = 50
arange = [2, 3, 4]
plt.figure(figsize=(10,6))
plt.title("Gamma functions")
plt.xlabel("x")
plt.ylabel("$\Gamma(a)$")

for a in arange:
    y = gamma(xrange, a)
    plt.plot(xrange, y, label=f"$\Gamma({a})$")

plt.legend()
plt.savefig("1a.png")
plt.show()


# In[5]:


def gamma_new(z, a):
    return np.exp(-(z*0.5)/(1-z)+(a-1)*np.log((z*0.5)/(1-z)))*(0.5+(z*0.5)/(1-z))**2/0.5


# In[6]:


def gauss_quad_new(p,q,N,a):
    x, w = gaussxwab(N,p,q)
    G = 0
    for k in range(N):
        G += w[k]*gamma_new(x[k], a)
    return G


# In[7]:


gammanew = gauss_quad_new(0,1,100,3/2)
print('The gamma value desired for this part is',gammanew)


# In[8]:


for a in [2, 6, 10]:
    print('The gamma value desired for gamma',a,'is',gauss_quad_new(0,1,10000,a))


# In[ ]:




