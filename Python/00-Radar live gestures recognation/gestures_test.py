import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.layers import Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D




################ PARAMETERS ########################

path_test='Test/'

test=[]

t_label=[]

sweeps=45
samples=207




################## Data Retrive and Assign Labels 0 and 1 ########
Swipe_R_to_L_RightHand = os.listdir(path_test + 'Swipe_R_to_L_RightHand/')
for i, gestures in enumerate(Swipe_R_to_L_RightHand):    #Remember enumerate method adds a counter and returns the enumerate object
    
    if (gestures.split('.')[1] == 'npy'):
         gesture= np.load(path_test + 'Swipe_R_to_L_RightHand/' + gestures)
         gesture = gesture.reshape((sweeps, samples))
         gesture = np.swapaxes(gesture,0,1)
         gesture = np.abs(np.fft.fftshift(np.fft.fft(gesture),axes=(1,)))
         test.append(np.array(gesture))
         t_label.append(0)
         
         
         
ZoomOut_RightHand    = os.listdir(path_test + 'ZoomOut_RightHand/')
for i, gestures in enumerate(ZoomOut_RightHand   ):    #Remember enumerate method adds a counter and returns the enumerate object
    
    if (gestures.split('.')[1] == 'npy'):
         gesture= np.load(path_test + 'ZoomOut_RightHand/' + gestures)
         gesture = gesture.reshape((sweeps, samples))
         gesture = np.swapaxes(gesture,0,1)
         gesture = np.abs(np.fft.fftshift(np.fft.fft(gesture),axes=(1,)))
         test.append(np.array(gesture))
         t_label.append(1) 
         
         
###################### npy array ################      
test = np.array(test)
test=np.expand_dims(test,axis=-1)
t_label = np.array(t_label)
print(test.shape)
print(t_label.shape)
prediction=[]
################# Test #########
from tensorflow import keras

model = keras.models.load_model('gestures.h5')


n= 45  #Select the index of image to be loaded for testing
for i in range(len(t_label)):
    ges = test[i]

    input_ges= np.expand_dims(ges, axis=0) #Expand dims so the input is (num ges,samples, sweeps, 1)
    print(input_ges.shape)
    print("The prediction for this gesture is: ",np.argmax( model.predict(input_ges)))
    print("The actual label for this gesture is: ",t_label[i])