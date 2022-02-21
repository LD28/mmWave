
import asyncio
import asyncio
import websockets
import numpy as np
from tensorflow import keras
import time
length=99820
data_buffer=np.zeros(length)
count=0
samples=620
import threading
pr_count=0
async def echo(websocket):
    global data_buffer
    global count
    count2=0
    
    global pr_count
    async for message in websocket:
        pr_count=pr_count+1
        string=message[1:-1]
        string=string.split(',')
        newlist=[int(element) for element in string]
        temp=data_buffer[:-samples]
        data_buffer[:samples]=newlist
        data_buffer[samples:]=temp
        print('receiving')
        count2=count2+1
        if count2>80:
            gesture_prediction()
            count2=0
            
        
        

        count=count+1
        print('count is ', count)
model = keras.models.load_model('/home/pi/Acconeer/smart_mirror/final.h5')
def gesture_prediction():
    global pr_count
    while True:
        
        
        global data_buffer
        
        print('prediction thread')
        if pr_count>80:
            gesture = data_buffer.reshape((161, 620))

            gesture = np.expand_dims(gesture, axis=0)
            start = time.time()
            model.predict(gesture)
           
            stop = time.time()
            print(stop-start)
            pr_count=0


async def main():
    async with websockets.serve(echo, "localhost",8133,ping_interval=None):
        await asyncio.Future()  # run fore
#model_thread = threading.Thread(target=gesture_prediction)


# model_thread = threading.Thread(target=gesture_prediction)
# model_thread.setDaemon(True)
# model_thread.start()   
asyncio.run(main())
