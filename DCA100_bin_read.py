'''
RadarCfg
'''
Cfg = {'start_frequency': 76e9,
            'idle': 100e-6,
            'adcStartTime': 6e-6,
            'rampEndTime': 60e-6,
            'txStartTime': 0,
            'adcSamples': 153,
            'adcSampleRate': 5000e3,
            'freqSlopeConst': 80.896e6/1e-6,
            'txPower': 12,
            'txPhaseShift': 0,
            'hpfCornerFreq1': 175e3,
            'hpfCornerFreq2': 350,
            'rxGain': 30,
            'numChirps': 128,
            'numLanes': 4,
            'isComplex': True,
            'c' : 2.998e8
           }


'''
labraries
'''
from matplotlib import pyplot as plt
import numpy as np
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
    
print(adcData)


