#!/usr/bin/env python
# coding: utf-8

# In[15]:


import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize


# In[2]:


def f(x):
    return (x-0.3)**2*np.exp(x)


# In[3]:


def s_quad_interp(f,a, b, c):
    """
    inverse quadratic interpolation
    """
    epsilon = 1e-7 #for numerical stability
    s0 = a*f(b)*f(c) / (epsilon + (f(a)-f(b))*(f(a)-f(c)))
    s1 = b*f(a)*f(c) / (epsilon + (f(b)-f(a))*(f(b)-f(c)))
    s2 = c*f(a)*f(b) / (epsilon + (f(c)-f(a))*(f(c)-f(b)))
    return s0+s1+s2


# In[25]:


def golden(func=f,astart=None, bstart=None, cstart=None, tol=1.e-5):
    a = astart
    b = bstart
    c = cstart
    itercount = 0
    while(np.abs(c - a) > tol) & (itercount<300):
        # Split the larger interval
        if((b - a) > (c - b)):
            x = b
            b = b - gsection * (b - a)
        else:
            x = b + gsection * (c - b)
        step = np.array([b, x])
        fb = func(b)
        fx = func(x)
        if(fb < fx):
            c = x
        else:
            a = b
            b = x
        itercount+=1
    return (b)


# In[7]:


def optimizeg(a,b,c):
    #define interval
    tol = 1e-7
    if abs(f(a)) < abs(f(b)):
        a, b = b, a #swap bounds
    c = a
    flag = True
    err = abs(b-a)
    err_list, b_list = [err], [b]
    while err > tol:
        s = s_quad_interp(a,b,c)
        if ((s >= b))            or ((flag == True) and (abs(s-b) >= abs(b-c)))            or ((flag == False) and (abs(s-b) >= abs(c-d))):
            s = (a+b)/2 #bisection
            flag = True
        else:
            flag = False
        c, d = b, c # d is c from previous step
        #if f(a)*f(s) < 0:
        #    b = s
        #else:
        a = s
        if abs(f(a)) < abs(f(b)):
            a, b = b, a #swap if needed
        err = abs(b-a) #update error to check for convergence
        err_list.append(err)
        b_list.append(b)
    return s


# In[30]:


optimized = optimizeg(-1,0.5,1)


# In[32]:


minimized = scipy.optimize.brent(f, brack=(-1, 0.5, 1), tol=1.0e-5)


# In[33]:


print(optimized)
print(minimized)


# In[34]:


print('The difference between my optimization with scipy package is',abs(minimized -optimized))


# In[ ]:




