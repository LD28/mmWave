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
from matplotlib import pyplot as plt
from scipy.fft import fft, ifft
import numpy as np
'''
Read .bin generted from mmwave-studtio DCA-1000
'''
numADCBits = 16# % number of ADC bits per sample
numLanes = 4#% do not change. number of lanes is always 4 even if only 1 lane is used. unused
isReal = False# set to 1 if real only data, 0 if complex dataare populated with 0 %% read file and
# DCA1000 should read in two's complement data
adcData= np.fromfile('23sep/2.bin',dtype=np.int16)
if (numADCBits != 16):
    l_max = 2^(numADCBits-1)-1
    adcData = adcData(adcData > l_max) - 2^numADCBits
    
if (isReal==True):
    adcData = np.reshape(adcData, numLanes, [])
else:
    adcData = adcData.reshape(-1, numLanes*2)
    adcData= adcData[:, :4] + 1j* adcData[:, 4:]

    
    

#plot_range_time of single chirp

singleChrip=adcData[0:numADCSamples,0]

fft_=fft(singleChrip,numADCSamples)

max_=np.max(fft_)
normalizedfft=fft_/max_
absN=np.abs(normalizedfft)

bin_ =np.arange(0, numADCSamples, 1)
fdel_bin=bin_*(SampleRate/numADCSamples)

d=numADCSamples*fdel_bin/freqSlopeConst

# Get sample points for the discrete signal(which represents a continous signal)
x=d
y = absN
# plot the signal in time domain
plt.figure(figsize = (8, 6))
plt.plot(x,y,'g')
plt.xlabel('Ragne(m)')
plt.ylabel('Normlized Amplitude')
plt.show()

























