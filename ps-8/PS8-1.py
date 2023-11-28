#!/usr/bin/env python
# coding: utf-8

# In[54]:


import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq,irfft,rfft,rfftfreq
from scipy.integrate import solve_ivp


# In[43]:


import pandas as pd

piano = pd.read_csv('piano.txt', header = None).to_numpy()
trumpet = pd.read_csv('trumpet.txt', header = None).to_numpy()


# In[44]:


plt.plot(np.arange(0, len(piano)), piano)
plt.xlabel('time')
plt.ylabel('amplitude')
plt.title('Piano Waveform')
plt.savefig('Waveform_P.png')


# In[60]:



plt.plot(np.arange(0, len(trumpet)),trumpet)
plt.xlabel('time')
plt.ylabel('amplitude')
plt.title('Trumpet Waveform')
plt.savefig('Waveform_T.png')


# In[63]:


Np=len(piano)
Nt=len(trumpet)
T=1/44100
piano = piano.transpose().reshape(100000)
trumpet = trumpet.transpose().reshape(100000)
piano_fft = rfft(piano)
pianofreq = rfftfreq(Np,d=T)
plt.plot(pianofreq, np.abs(piano_fft))
plt.xlim(0,10000)
plt.xlabel('Frequency,(Hz)')
plt.ylabel('Amplitude')
plt.title('Piano first 10000')
plt.savefig('PianoFFT.png')


# In[64]:


trumpet_fft = rfft(trumpet)
trumpetfreq = rfftfreq(Nt,d=T)
plt.plot(pianofreq, np.abs(trumpet_fft))
plt.xlim(0,10000)
plt.xlabel('Frequency,(Hz)')
plt.ylabel('Amplitude')
plt.title('Trumpet first 10000')
plt.savefig('TrumpetFFT.png')


# In[76]:


max_piano_index = np.where(piano_fft == np.max(piano_fft))
piano_frequency = pianofreq[max_piano_index]
print('Most frequency piano is', piano_freq[max_piano_index],'Hz')


# In[75]:


max_trumpet_index = np.where(trumpet_fft == np.max(trumpet_fft))
trumpet_frequency = trumpetfreq[max_trumpet_index]

print('Most frequency of trumpet is', trumpet_frequency,'Hz')


# In[ ]:




