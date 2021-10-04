'''
RadarCfg
'''
numADCBits= 16# number of ADC bits per sample.
numADCSamples= 151# number of ADC samples per chirp.
numRx= 4# number of receivers.
numChirps= 128
chirpSize= numADCSamples*numRx
freqSlopeConst= 80.896e6/1e-6
chirploops=128# No. of of chirp loops. 
numLanes=4# do not change. number of lanes is always 4.
isComplex=True#set to False if real only data, True if complex data.
numFrames=100
frameTime= 0.100# 40ms per frame
totalTime= frameTime*numFrames
SampleRate=5000e3
timeStep= 1/SampleRate# [us]
chirpPeriod= numADCSamples * timeStep# [us]
plotEnd=numADCSamples * numChirps*numFrames
timeEnd= (plotEnd-1) * timeStep
c=2.998e8
      


'''
labraries
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fftpack import fft
'''
Read .bin generted from mmwave-studtio DCA-1000
'''
numADCBits = 16# % number of ADC bits per sample
numLanes = 4#% do not change. number of lanes is always 4 even if only 1 lane is used. unused
isReal = False# set to 1 if real only data, 0 if complex dataare populated with 0 %% read file and
# DCA1000 should read in two's complement data
adcData= np.fromfile('23sep/1.6.bin',dtype=np.int16)
if (numADCBits != 16):
    l_max = 2^(numADCBits-1)-1
    adcData = adcData(adcData > l_max) - 2^numADCBits
    
if (isReal==True):
    adcData = np.reshape(adcData, numLanes, [])
else:
    adcData = adcData.reshape(-1, numLanes*2)
    adcData= adcData[:, :4] + 1j* adcData[:, 4:]

##############################################################################


bin_ =np.arange(0, numADCSamples, 1)
fdel_bin=bin_*(SampleRate/numADCSamples)
distance=numADCSamples*fdel_bin/freqSlopeConst
##############################################################################
    
rx1= adcData[:, 0]
rx2= adcData[:, 1]
rx3= adcData[:, 2]
rx4= adcData[:, 3]

#Reshape the Chenals
rx1 = rx1.reshape(15100,128)
rx2 = rx2.reshape(15100,128)
rx3 = rx3.reshape(15100,128)
rx4 = rx4.reshape(15100,128)

#Take the fft of RX
signal_fft_1 = fft(rx1, numADCSamples)
signal_fft_2 = fft(rx2, numADCSamples)
signal_fft_3 = fft(rx3, numADCSamples)
signal_fft_4 = fft(rx4, numADCSamples)

#Take the absolute value of FFT output
signal_fft_1 = np.abs(signal_fft_1)/np.max(np.abs(signal_fft_1))
signal_fft_2 = np.abs(signal_fft_2)/np.max(np.abs(signal_fft_2))
signal_fft_3 = np.abs(signal_fft_3)/np.max(np.abs(signal_fft_3))
signal_fft_4 = np.abs(signal_fft_4)/np.max(np.abs(signal_fft_4))

# plot the signal in time domain
x=distance
y1=signal_fft_1 
y2=signal_fft_2
y3=signal_fft_3 
y4=signal_fft_4 
y1=np.swapaxes(y1, 0, 1)
y2=np.swapaxes(y2, 0, 1)
y3=np.swapaxes(y3, 0, 1)
y4=np.swapaxes(y4, 0, 1)

#plotting the range
plt.figure(figsize = (8, 6))
plt.plot(x,y1)
plt.plot(x,y2)
plt.plot(x,y3)
plt.plot(x,y4)
plt.xlabel('Ragne(m)')
plt.ylabel('Normlized Amplitude')
plt.show()

##############################################################################


               











