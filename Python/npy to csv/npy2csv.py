
import os

import numpy as np

################ PARAMETERS ########################
path = 'G4_dataset/'



sweeps=50
samples=207

############## DARA

swipe = os.listdir(path + 'swipe/')
for i, gestures in enumerate(swipe):    #Remember enumerate method adds a counter and returns the enumerate object
    
    if (gestures.split('.')[1] == 'npy'):
         gesture= np.load(path + 'swipe/' + gestures)
         gesture = gesture.reshape((sweeps,samples))
         np.savetxt(( str(i) +".csv"), gesture, delimiter=",")
 