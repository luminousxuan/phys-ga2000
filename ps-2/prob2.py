#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import timeit


# In[24]:


# We can derive the equation firstly. See the \epsilon_0, e and a are gone in the Madelung constant.
def madelung_const_forloop(number):
    M = 0 #Madelung constant
    L = number
    for i in range(-L,L+1):
        for j in range(-L,L+1):
            for k in range(-L,L+1):
                if i==0 and j==0 and k==0:
                    continue
                else:
                    M += (-1)**(i+j+k)/(np.sqrt(i**2+j**2+k**2))
    return M


# In[34]:


get_ipython().run_line_magic('timeit', '-n 1 print (madelung_const_forloop(100))')


# In[51]:


def madelung_const_noforloop(number):
    L=number
    nums = np.arange(-L,L+1,dtype=np.float64)
    i, j, k = np.meshgrid(nums, nums, nums)
    m = (-1)**(i + j + k)/np.sqrt(i**2 + j**2 + k**2)
    m[(i == 0)*(j == 0)*(k == 0)] = 0
    M = np.sum(m)
    return M


# In[52]:


get_ipython().run_line_magic('timeit', '-n 1 print (madelung_const_noforloop(100))')


# In[ ]:




