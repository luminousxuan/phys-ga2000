#!/usr/bin/env python
# coding: utf-8

# In[110]:


import numpy as np
import matplotlib.pyplot as plt
import scipy
import copy
from banded import banded
from scipy import linalg


# In[111]:


L = 1e-8
N = 1000
a = L/N
x0 = L/2
sigma = 1e-10
k = 5e10
m = 9.109e-31
h = 1e-18
hbar = 1.0545718e-34


# In[112]:


a1 = 1 + h * 1j * hbar / (2*m*a**2)
a2 = -h * 1j * hbar / (4*m*a**2)
b1 = 2 - a1
b2 = -a2


# In[113]:


A = np.empty((3,N),complex)
A[0,:] = a2
A[1,:] = a1
A[2:,] = a2


# In[114]:


N = 1001
A_diag = np.ones(N, dtype=complex)*a1
A_u = np.ones(N, dtype=complex) * a2
A_u[0] = 0
A_l = np.ones(N, dtype=complex) * a2
A_l[-1] = 0
A = np.array([A_u, A_diag, A_l])


# In[115]:


def psi0(x):
    return np.exp(-(x-x0)**2/2/sigma**2)*np.exp(1j*k*x)
x = np.linspace(0,L,N)
psi[:] = psi0(x)
psi[[0,N-1]]=0


# In[126]:


psi_list = []
for t in range(10000):
    psi_list.append(psi)
    psiold = psi
    psiold = np.concatenate(([0],psi,[0])) 
    v = b1*psiold[1:-1] + b2*(psiold[2:]+psiold[:-2])
    psi = linalg.solve_banded((1,1), A, v)
    #Copy can't be imported so I didn't use given banded module
    psi[0] = psi[-1] = 0
psi_values = np.array(psi_values, dtype=complex)
real = np.real(psi_values)


# In[133]:


plt.plot(x, real[0], label='Psi(t=0)')
plt.plot(x, real[100], label='Psi(t=0.01)')
plt.plot(x, real[200], label='Psi(t=0.02)')
plt.plot(x, real[300], label='Psi(t=0.03)')
plt.plot(x, real[400], label='Psi(t=0.04)')
plt.plot(x, real[1000], label='Psi(t=0.10)')
plt.plot(x, real[9900], label='Psi(t=0.99)')
plt.xlabel('Position ($1x10^{-8}$m)')
plt.ylabel('Amplitude')
plt.ylim(-1.1,1.1)
plt.legend()


# In[ ]:




