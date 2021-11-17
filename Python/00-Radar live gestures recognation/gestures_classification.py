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
path = 'Dataset/'
dataset= []
classes= []
label= []
classes= os.listdir(path)
sweeps=45
samples=207




################## Data Retrive and Assign Labels 0 and 1 ########
Swipe_L_to_Ri_RightHand = os.listdir(path + 'Swipe_L_to_Ri_RightHand/')
for i, gestures in enumerate(Swipe_L_to_Ri_RightHand):    #Remember enumerate method adds a counter and returns the enumerate object
    
    if (gestures.split('.')[1] == 'npy'):
         gesture= np.load(path + 'Swipe_L_to_Ri_RightHand/' + gestures)
         gesture = gesture.reshape((sweeps, samples))
         gesture = np.swapaxes(gesture,0,1)
         gesture = np.abs(np.fft.fftshift(np.fft.fft(gesture),axes=(1,)))
         dataset.append(np.array(gesture))
         label.append(0)
         
         
         
         
ZoomOut_RightHand = os.listdir(path + 'ZoomOut_RightHand/')
for i, gestures in enumerate(ZoomOut_RightHand):    #Remember enumerate method adds a counter and returns the enumerate object
    
    if (gestures.split('.')[1] == 'npy'):
         gesture= np.load(path + 'ZoomOut_RightHand/' + gestures)
         gesture = gesture.reshape((sweeps, samples))
         gesture = np.swapaxes(gesture,0,1)
         gesture = np.abs(np.fft.fftshift(np.fft.fft(gesture),axes=(1,)))
         dataset.append(np.array(gesture))
         label.append(1)
         
         

No_Gesture = os.listdir(path + 'No_Gesture/')
for i, gestures in enumerate(No_Gesture):    #Remember enumerate method adds a counter and returns the enumerate object
    
    if (gestures.split('.')[1] == 'npy'):
         gesture= np.load(path + 'No_Gesture/' + gestures)
         gesture = gesture.reshape((sweeps, samples))
         gesture = np.swapaxes(gesture,0,1)
         gesture = np.abs(np.fft.fftshift(np.fft.fft(gesture),axes=(1,)))
         dataset.append(np.array(gesture))
         label.append(2)
                   
         
         
###################### npy array ################         
dataset = np.array(dataset)
classes = np.array(classes)
dataset=np.expand_dims(dataset,axis=-1)

print(dataset.shape)
print(classes.shape)

############## split into train and test ###############

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dataset,label, test_size = 0.20, random_state = 0)

############## togetagorical ######

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train, len(classes))
y_test = to_categorical(y_test, len(classes))
############## CNN Model ########
INPUT_SHAPE = (samples,sweeps,1)   #change to (45, 207,1)

from keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as k



model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape = INPUT_SHAPE, activation='relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(128, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))





#Do not use softmax for binary classification
#Softmax is useful for mutually exclusive classes, either cat or dog but not both.
#Also, softmax outputs all add to 1. So good for multi class problems where each
#class is given a probability and all add to 1. Highest one wins. 

#Sigmoid outputs probability. Can be used for non-mutually exclusive problems.
#But, also good for binary mutually exclusive (cat or not cat). 

model.compile(loss='binary_crossentropy',
              optimizer='adam',             #also try adam,rmsprop
              metrics=['accuracy'])

print(model.summary())    
############################### Training #################

history = model.fit(X_train, 
                         y_train, 
                         batch_size = 64, 
                         verbose = 1, 
                         epochs = 200,      
                         validation_data=(X_test,y_test),
                         shuffle = False
                     )


model.save('gestures.h5') 


#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

'''
################# Test #########

n=25  #Select the index of gesture to be loaded for testing
gesture_ = X_test[n]
plt.imshow(gesture_)
input_ges = np.expand_dims(gesture_, axis=0) #Expand dims so the input is (num images, x, y, c)
print("The prediction for this gesture is: ", model.predict(input_ges))
print("The actual label for this gesture is: ", y_test[n])
'''
