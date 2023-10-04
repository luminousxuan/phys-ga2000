#!/usr/bin/env python
# coding: utf-8

# In[27]:


#Scipy this time
import numpy as np
from scipy.special import roots_hermite
import math
from numpy import ones,copy,cos,tan,pi,linspace
import matplotlib.pyplot as plt


# In[10]:


def Hermite(n,x):
    if n==0:
        return np.ones(x.shape)
    elif n == 1:
        return 2*x
    else:
        return 2*x*Hermite(n-1,x)-2*(n-1)*Hermite(n-2,x)
def psi(n, x):
    return (1 / np.sqrt(2**n * math.factorial(n) * np.sqrt(np.pi))) * np.exp(-x**2 / 2) * Hermite(n, x)


# In[18]:


xlist = np.linspace(-4,4,100)
for i in np.arange(0,4):
    lab = str(i)
    plt.plot(xlist,psi(i,xlist),label = "n="+lab)
plt.legend()
plt.title('Harmonic Oscillator Wavefunctions')
plt.xlabel('x')
plt.ylabel('Psi(x)')
plt.savefig('HO_wf.jpg')
plt.show()


# In[22]:


xlist2 = np.linspace(-10,10,250)
psib = psi(30,xlist2)
plt.plot(xlist2,psib)
plt.title('Harmonic Oscillator for n=30 case')
plt.xlabel('x')
plt.ylabel('Psi(x)')
plt.savefig('HO_n=30.png')
plt.show()


# In[23]:


def integrand(x):
    k = 1-x**2
    return ((1 + x**2) / k**2) * (x / k)**2 * np.abs(psi(5, (x / k)))**2


# In[28]:


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


# In[33]:


def gaussmethod(a, b, N):
    x, w = gaussxwab(N,a,b)
    s = sum(integrand(x)*w)
    return s
integral = gaussmethod(-1, 1, 100)
print("The quantum uncertaity should be" ,np.sqrt(integral))


# In[40]:


def integrandnew(x):
    return x * x / (pow(2, 5)* math.factorial(5) * math.sqrt(math.pi)) * pow(Hermite(5, x), 2)
xnew, wnew = roots_hermite(100)
fnew = integrandnew(xnew)
uncer = math.sqrt(np.sum(wnew * fnew))
print("The quantum uncertaity should be",uncer)

