# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 23:46:22 2021

@author: RIMMS-RADAR
"""
#RadarCfg

numADCBits = 16# % number of ADC bits per sample
isReal = False# set to 1 if real only data, 0 if complex dataare populated with 0 %% read file and
numADCSamples= 128# number of ADC samples per chirp.
num_rx= 4# number of receivers.
num_tx=2
numChirps= 128
chirpSize= numADCSamples*num_rx
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
num_vx = num_tx * num_rx # Number of virtual antennas


# Imports
import numpy as np                  # Scientific Computing Library
import matplotlib.pyplot as plt     # Basic Visualization Library

# Read in frame data
'''
Read .bin generted from mmwave-studtio DCA-1000
'''
# DCA1000 should read in two's complement data
adcData= np.fromfile('1m.bin',dtype=np.int16)
if (numADCBits != 16):
    l_max = 2^(numADCBits-1)-1
    adcData = adcData(adcData > l_max) - 2^numADCBits
    
if (isReal==True):
    adcData = np.reshape(adcData, num_rx, [])
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
##############################################################################
all_frame =np.zeros((rx1_allchirps.shape[0],num_vx,rx1_allchirps.shape[1]), dtype=complex) 

for x in range(rx1_allchirps.shape[0]): 
    all_frame[x,:]= rx1_allchirps[x,:]
    

frame1= all_frame[1280:1440,:,:]

  






range_plot = np.fft.fft(frame1, axis=2)
'''
# Visualize Results
plt.imshow(np.abs(range_plot.sum(1)).T)
plt.ylabel('Range Bins')
plt.title('Interpreting a Single Frame - Range')
plt.show()
'''

range_doppler = np.fft.fft(range_plot, axis=0)
range_doppler = np.fft.fftshift(range_doppler, axes=0)
'''
# Visualize Results
plt.imshow(np.log(np.abs(range_doppler).T).sum(1))
plt.xlabel('Doppler Bins')
plt.ylabel('Range Bins')
plt.title('Interpreting a Single Frame - Doppler')
plt.show()
'''
#################### TODO ####################
"""
    Task: Perform the range FFt on the entire frame
    Output: range_azimuth [azimuth_bins, range_bins]
    
    Details: There are three basic things you will need to do here:
                1. Zero pad each virtual antenna array (axis 1) from 8 elements to num_angle_bins elements
                2. Perform the Azimuth FFT
                3. Accumlate result over all doppler bins (for visualization purposes)
"""
num_angle_bins = 64

# Zero pad input

# Azimuth FFT

# Accumulate over all doppler bins

##############################################
padding = ((0,0), (0,num_angle_bins-range_doppler.shape[1]), (0,0))
range_azimuth = np.pad(range_doppler, padding, mode='constant')
range_azimuth = np.fft.fft(range_azimuth, axis=1)

# Visualize Results
plt.imshow(np.log(np.abs(range_azimuth).sum(1).T))
plt.xlabel('Azimuth (Angle) Bins')
plt.ylabel('Range Bins')
plt.title('Interpreting a Single Frame - Azimuth')
plt.show()

