#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


N = 1000
tau = 3.053*60

t_decay = -1/(np.log(2)/tau)*np.log(1-np.random.random(N))
t_decay = np.sort(t_decay)
decayed = np.arange(1,N+1)
survived = -decayed + N

plt.plot(t_decay,survived,label='survived atom')
plt.plot(t_decay,decayed,label='decayed atom')
plt.legend()
plt.xlabel('Time,s')
plt.ylabel('Number of Atoms')
plt.title('Decay to Tl208')
plt.savefig('DecaytoTl208_p4')
plt.show()


# In[ ]:




