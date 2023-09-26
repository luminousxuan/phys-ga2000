#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import matplotlib.pyplot as plt
import time


# In[24]:


def multiplication_loop(n,A,B):
    C = np.zeros([n,n], float)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i,j] += A[i,k] * B[k,j]
    return C

def multiplication_dot(n,A,B):
    return np.dot(A,B)


# In[34]:


runrange=np.arange(20,400,20)
t_loop=[]
t_dot=[]
for i in runrange:
    at0 = time.time()
    multiplication_loop(i,np.random.random((i,i)),np.random.random((i,i)))
    at1 = time.time()
    t_loop.append(at1-at0)
    at2 = time.time()
    multiplication_dot(i,np.random.random((i,i)),np.random.random((i,i)))
    at3 = time.time()
    t_dot.append(at3-at2)


# In[39]:


plt.plot(t_loop, label="for loops")
plt.plot(t_dot, label="dot method")
plt.legend()
plt.xlabel('Matrix size for one dimension,20*(N+1)')
plt.ylabel('Computation Time,s')
plt.title('matrix multiplication computational time by methods')
plt.savefig("matrix_multiplication.png")
plt.show()


# In[ ]:




