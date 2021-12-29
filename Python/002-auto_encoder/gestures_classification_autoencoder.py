
import os
import tensorflow as tf
from tensorflow import keras
import warnings
import numpy as np
import cv2

from sklearn import metrics
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Input, Dropout
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Model
import seaborn as sns

from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv1D, MaxPool1D, Dropout, LSTM
from tensorflow.keras.optimizers import RMSprop, Adam, SGD, Adagrad, Adadelta
from keras.metrics import categorical_crossentropy

from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint, EarlyStopping
warnings.simplefilter(action='ignore', category=FutureWarning)

from keras.regularizers import l2


import matplotlib.pyplot as plt
################ PARAMETERS ########################
path = 'G7_dataset/'
print (path)
x_gesture= []
x_no_gesture= []

sweeps=50
samples=207

############## 
      

         
button_press = os.listdir(path + 'button_press/')
for i, gestures in enumerate(button_press):    #Remember enumerate method adds a counter and returns the enumerate object
    
    if (gestures.split('.')[1] == 'npy'):
         gesture= np.load(path + 'button_press/' + gestures)
         gesture = gesture.reshape((sweeps,samples))
         gmax=np.max(gesture)
         gesture =gesture/gmax 
         x_gesture.append(np.array(gesture))
         
         
         

finger_slide = os.listdir(path + 'finger_slide/')
for i, gestures in enumerate(finger_slide):    #Remember enumerate method adds a counter and returns the enumerate object
    
    if (gestures.split('.')[1] == 'npy'):
         gesture= np.load(path + 'finger_slide/' + gestures)
         gesture = gesture.reshape((sweeps,samples))
         gmax=np.max(gesture)
         gesture =gesture/gmax
         x_gesture.append(np.array(gesture))
         

hand_away = os.listdir(path + 'hand_away/')
for i, gestures in enumerate(hand_away):    #Remember enumerate method adds a counter and returns the enumerate object
    
    if (gestures.split('.')[1] == 'npy'):
         gesture= np.load(path + 'hand_away/' + gestures)
         gesture = gesture.reshape((sweeps,samples))
         gmax=np.max(gesture)
         gesture =gesture/gmax
         x_gesture.append(np.array(gesture))
         

hand_closer = os.listdir(path + 'hand_closer/')
for i, gestures in enumerate(hand_closer):    #Remember enumerate method adds a counter and returns the enumerate object
    
    if (gestures.split('.')[1] == 'npy'):
         gesture= np.load(path + 'hand_closer/' + gestures)
         gesture = gesture.reshape((sweeps,samples))
         gmax=np.max(gesture)
         gesture =gesture/gmax
         x_gesture.append(np.array(gesture))
         

swipe_left = os.listdir(path + 'swipe_left/')
for i, gestures in enumerate(swipe_left):    #Remember enumerate method adds a counter and returns the enumerate object
    
    if (gestures.split('.')[1] == 'npy'):
         gesture= np.load(path + 'swipe_left/' + gestures)
         gesture = gesture.reshape((sweeps,samples))
         gmax=np.max(gesture)
         gesture =gesture/gmax
         x_gesture.append(np.array(gesture))
         
swipe_right = os.listdir(path + 'swipe_right/')
for i, gestures in enumerate(swipe_right):    #Remember enumerate method adds a counter and returns the enumerate object
    
    if (gestures.split('.')[1] == 'npy'):
         gesture= np.load(path + 'swipe_right/' + gestures)
         gesture = gesture.reshape((sweeps,samples))
         gmax=np.max(gesture)
         gesture =gesture/gmax
         x_gesture.append(np.array(gesture))
                  

no_gesture = os.listdir(path + 'no_gesture/')
for i, gestures in enumerate(no_gesture):    #Remember enumerate method adds a counter and returns the enumerate object
    
    if (gestures.split('.')[1] == 'npy'):
         gesture= np.load(path + 'no_gesture/' + gestures)
         gesture = gesture.reshape((sweeps,samples))
         gmax=np.max(gesture)
         gesture =gesture/gmax
         x_no_gesture.append(np.array(gesture))
                              
         
###################### npy array ################         
x_gesture = np.array(x_gesture)
x_no_gesture = np.array(x_no_gesture)


print(x_gesture.shape)
print(x_no_gesture.shape)


############## split into train and test ###############

from sklearn.model_selection import train_test_split



x_gesture_train, x_gesture_test = train_test_split(
        x_gesture, test_size=0.20, random_state=42)


encoder_input = keras.Input(shape=(sweeps,samples), name='ges')
print(encoder_input)

x = keras.layers.Flatten()(encoder_input)
print(x)
encoder_output = keras.layers.Dense(64, activation="relu")(x)

print(encoder_output)

encoder = keras.Model(encoder_input, encoder_output, name='encoder')

print(encoder)

decoder_input = keras.layers.Dense(64, activation="relu")(encoder_output)
print(decoder_input)

x = keras.layers.Dense((sweeps*samples), activation="relu")(decoder_input)
print(x)

decoder_output = keras.layers.Reshape((50,207))(x)

print(decoder_output)

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
print(opt)

autoencoder = keras.Model(encoder_input, decoder_output, name='autoencoder')
print(autoencoder )

autoencoder.summary()

autoencoder.compile(opt, loss='mse')


history = autoencoder.fit(x_gesture_train,
  x_gesture_train,
  epochs=100, 
  batch_size=32, validation_split=0.10
)   
autoencoder.save("ae.h5")



plt.plot(history.history['loss'], label='Training loss')

plt.show()

plt.plot(history.history['val_loss'], label='Validation loss')

plt.show()




pred1 = autoencoder.predict(x_gesture_test)

pred1 = pred1[30]
score2 = np.sqrt(metrics.mean_squared_error(pred1,x_no_gesture[30]))

print(score2)










pred2 = autoencoder.predict(x_gesture)

pred2 = pred2[30]
score2 = np.sqrt(metrics.mean_squared_error(pred2,x_gesture[30]))

print(score2)




trainPredict = autoencoder.predict(x_no_gesture)
trainMAE = np.mean(np.abs(trainPredict - x_no_gesture), axis=1)
plt.hist(trainMAE, bins=30)
plt.show()


testPredict = autoencoder.predict(x_gesture_test)
testMAE = np.mean(np.abs(testPredict - x_gesture_test), axis=1)
plt.hist(testMAE, bins=30)
plt.show()



#######################













