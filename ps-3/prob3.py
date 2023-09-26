#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
from random import random
random()
# Initial

h=1.0

NBi209 = 0
NPb = 0
NTi = 0
NBi213 = 10000



probPb = 1 - 2**(-h/(3.3*60))
probTi = 1 - 2**(-h/(2.2*60))
probBi = 1 - 2**(-h/(46*60))

Bi209_point = []
Bi213_point = []
Pb_point = []
Ti_point = []


t = np.arange(0,20000,h)
for ti in t:
    Bi209_point.append(NBi209)
    Bi213_point.append(NBi213)
    Pb_point.append(NPb)
    Ti_point.append(NTi)
    
    
    for i in range(NPb):
        if random()<probPb:
            NPb-=1
            NBi209+=1
        
    for i in range(NTi):
        if random()<probTi:
            NTi-=1
            NPb+=1
        
    for i in range(NBi213):
        if random()<probBi:
            NBi213 -=1
            if random()>0.9791:
                NTi+=1
            else:
                NPb+=1
                
plt.plot(t,Bi209_point,label='Bi209')
plt.plot(t,Bi213_point,label='Bi213')
plt.plot(t,Pb_point,label='Pb209')
plt.plot(t,Ti_point,label='Ti209')

plt.title('Decay of Bi213')
plt.legend()
plt.xlabel('Time, s')
plt.ylabel('Number of atoms')
plt.savefig('DecayOfBi213_p3.png')
plt.show()


# In[ ]:




