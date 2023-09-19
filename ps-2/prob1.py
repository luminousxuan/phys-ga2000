#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np


# In[9]:


x = 100.98763
float_32 = np.float32(x)
int32bits = float_32.view(np.int32)
print("32bit for the given value is",'{:032b}'.format(int32bits))
bin32 = '{:032b}'.format(int32bits)
exponent = bin32[1:9]
mantissa = bin32[9:32]
print('exponet=',exponent)
print('mantissa=',mantissa)
diff = float_32 - x
print("The difference of the two expression is",diff)


# In[ ]:




