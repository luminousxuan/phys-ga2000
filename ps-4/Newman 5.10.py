#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import matplotlib.pyplot as plt
from numpy import ones,copy,cos,tan,pi,linspace

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


# In[11]:


def integrand(x, a):
    return 1 / (np.sqrt(a**4 - x**4))

def gquad(a, b, N, h):
    x, w = gaussxwab(N,a,b)
    s = sum(w*integrand(x,h))
    return s

def get_T(a):
    integral = gquad(0, a, 20, a)
    return np.sqrt(8)*integral

pointlist = np.linspace(0, 2, 50)
time = []
for a in pointlist:
    time.append(np.sqrt(8)*gquad(0,a,20,a))


# In[12]:


plt.plot(pointlist,time)
plt.title('Anharmonic Oscillator: V(x) = x^4')
plt.xlabel('Oscillation Amplitude (m)')
plt.ylabel('Time s')
plt.savefig('AhO.png')
plt.show()


# In[ ]:




