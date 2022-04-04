import numpy as np
from tensorflow import keras
import time
from HGR_at_3M_Baud_saaf import *
def gesture_prediction():
    #data_buffer = np.zeros(99820)
    model = keras.models.load_model('final.h5')
    count=0
    global data_buffer
    while True:
        global data_buffer
        count=count+1


        data_buffer=shared.value
        data = np.array(data_buffer)
        print(data)
        #data=np.array(data)
        print('prediction thread')
        gesture = data.reshape((161, 620))
            
        # gmax=np.max(gesture)
        # gesture =gesture/gmax
        gesture = np.expand_dims(gesture, axis=0)
        start = time.time()
        model.predict(gesture)
       
        stop = time.time()
        print(stop-start)

        
    
    
gesture_prediction()