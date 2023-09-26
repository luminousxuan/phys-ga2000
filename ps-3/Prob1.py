#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


def f(x):
    return (x*(x-1))


# In[4]:


def derivative(point,delta):
    return ((f(point+delta)-f(point))/delta)
    


# In[5]:


print(derivative( 1, 1e-2))


# In[6]:


print(derivative( 1, 1e-4))
print(derivative( 1, 1e-6))
print(derivative( 1, 1e-8))
print(derivative( 1, 1e-10))
print(derivative( 1, 1e-12))
print(derivative( 1, 1e-14))

