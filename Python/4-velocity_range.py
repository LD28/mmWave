# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 23:46:22 2021

@author: RIMMS-RADAR
"""
#RadarCfg

numADCBits = 16# % number of ADC bits per sample
isReal = False# set to 1 if real only data, 0 if complex dataare populated with 0 %% read file and
numADCSamples= 128# number of ADC samples per chirp.
numRx= 4# number of receivers.
num_tx=2
numChirps= 128
chirpSize= numADCSamples*numRx
freqSlopeConst= 82.993e6/1e-6
chirploops=128# No. of of chirp loops. 
numLanes=4# do not change. number of lanes is always 4.
isComplex=True#set to False if real only data, True if complex data.
numFrames=30
frameTime= 0.100# 40ms per frame
totalTime= frameTime*numFrames
SampleRate=5000e3
timeStep= 1/SampleRate# [us]
chirpPeriod= numADCSamples * timeStep# [us]
plotEnd=numADCSamples * numChirps*numFrames
timeEnd= (plotEnd-1) * timeStep
c=2.998e8
start_frequency= 76e9
idle=100e-6
rampEndTime= 60e-6


# Imports
import numpy as np                  # Scientific Computing Library
import matplotlib.pyplot as plt     # Basic Visualization Library

# Read in frame data
'''
Read .bin generted from mmwave-studtio DCA-1000
'''
# DCA1000 should read in two's complement data
adcData= np.fromfile('HandFB.bin',dtype=np.int16)
if (numADCBits != 16):
    l_max = 2^(numADCBits-1)-1
    adcData = adcData(adcData > l_max) - 2^numADCBits
    
if (isReal==True):
    adcData = np.reshape(adcData, numRx, [])
else:
    adcData = adcData.reshape(-1, numLanes*2)
    adcData= adcData[:, :4] + 1j* adcData[:, 4:]
        
##############################################################################
bin_ =np.arange(0, numADCSamples, 1)
fdel_bin=bin_*(SampleRate/numADCSamples)
distance=numADCSamples*fdel_bin/freqSlopeConst    

##############################################################################

rx1= adcData[:,0]
rx2= adcData[:,1]
rx3= adcData[:,2]
rx4= adcData[:,3]

rx1_allchirps =rx1.reshape(adcData.shape[0]//numADCSamples,numADCSamples)
rx2_allchirps =rx2.reshape(adcData.shape[0]//numADCSamples,numADCSamples)
rx3_allchirps =rx3.reshape(adcData.shape[0]//numADCSamples,numADCSamples)
rx4_allchirps =rx4.reshape(adcData.shape[0]//numADCSamples,numADCSamples)

#single_frame
single_frame=rx1_allchirps[3712:3840,:]
#np.save('single_frame.npy', single_frame)
##############################################################################

# Read in frame data
#frame = np.load('single_frame.npy')
frame=single_frame

# Manually cast to signed ints
frame.real = frame.real.astype(np.int16)
frame.imag = frame.imag.astype(np.int16)



# Range resolution
range_res = (c * SampleRate ) / (2 * freqSlopeConst  * numADCSamples)
print(f'Range Resolution: {range_res} [meters/second]')

# Apply the range resolution factor to the range indices
ranges = np.arange(numADCSamples) * range_res


# Make sure your equation translates to the following
velocity_res = c / (2 *start_frequency * (idle + rampEndTime) * numChirps * num_tx)
print(f'Velocity Resolution: {velocity_res} [meters/second]')

range_plot = np.fft.fft(frame, axis=1)

# Range FFT -> Doppler FFT
range_bins = np.fft.fft(frame, axis=1)
fft_2d = np.fft.fft(range_bins, axis=0)

# Doppler FFT -> Range FFT
doppler_bins = np.fft.fft(frame, axis=0)
rfft_2d = np.fft.fft(doppler_bins, axis=1)


# Take a sequential FFT across the chirps
range_doppler = np.fft.fft(range_plot, axis=0)

# FFT shift the values (explained later)
range_doppler = np.fft.fftshift(range_doppler, axes=0)


print('Max power difference: ', np.abs(fft_2d - rfft_2d).max())

# Apply the velocity resolution factor to the doppler indicies
velocities = np.arange(numChirps) - (numChirps // 2)
print(velocities)
velocities = velocities * velocity_res


powers = np.abs(range_doppler)

# Plot with units
plt.imshow(powers.T, extent=[velocities.min(), velocities.max(), ranges.max(), ranges.min()])
plt.xlabel('Velocity (meters per second)')
plt.ylabel('Range (meters)')
plt.show()

plt.plot(velocities, powers)
plt.xlabel('Velocity (meters per second)')
plt.ylabel('Reflected Power')
plt.title('Interpreting a Single Frame - Doppler')
plt.show()




