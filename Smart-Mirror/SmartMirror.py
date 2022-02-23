from tkinter import *
import tkinter.font
import time
from time import strftime
from datetime import date, timedelta
import pandas as pd

import datetime
import pickle
import os.path


WIDTH = 1024
HEIGHT = 1080

class GUI(Frame):

    def __init__(self, master):
        Frame.__init__(self, master)

        self.largeFont = tkinter.font.Font(family="Piboto", size=70)
        self.mediumFont = tkinter.font.Font(family="Piboto", size=40)
        self.normalFont = tkinter.font.Font(family="Piboto Light", size=20)

    def setupGUI(self):
        self.grid(row=0, column=0)

        # Weather & news frame to contain weather/news info
        # For weather, column 0 = info, column 1 = icon
        today_weather_frame = Frame(self, width=400, height=500, bg='black')
        today_weather_frame.grid(row=0, column=0, sticky=W)
        GUI.weather_label1 = Label(today_weather_frame, text="\nToday's weather:", fg='white', bg='black',
                                   font=self.mediumFont, justify=LEFT)
                                   
        GUI.weather_label1.grid(row=0, column=1, sticky=NW)
        
        # Frame and labels to hold the forecast
        
        
        weather_image = Frame(self, width=200, height=500, bg='black')
        weather_image.grid(row=1, column=0, sticky=W)
        GUI.weather_image = PhotoImage(file="smart_mirror_icons/weather.gif") 
        
        GUI.weather_image.grid(row=1, column=1, sticky=NW)
                                 
                                     
        
          
        
        
        # Set up labels to hold weather icons
        
      
      
      
  
       

        

        # Labels to hold news info
        news_frame = Frame(self, width=400, height=500, bg='black')
        news_frame.grid(row=2, column=0, sticky=W)

        GUI.news_today = Label(news_frame, text="\nToday's headlines:", fg='white', bg='black',
                               font=self.mediumFont, justify=LEFT)
        GUI.news_today.grid(row=0, column=0, sticky=W)

        


        # Adjust this width for spacing
        frame_placeholder = Frame(self, width=WIDTH/2.65, height=10, bg='black')
        frame_placeholder.grid(row=0, column=1)

        # Time frame to hold time & date in grid
        time_frame = Frame(self, width=400, height=500, bg='black')
        time_frame.grid(row=0, column=2, sticky=NE)
        GUI.time_label = Label(time_frame, text=strftime("%I:%M %p", time.localtime()), fg='white', bg='black',
                               font=self.largeFont)
        GUI.time_label.grid(row=0, column=0, sticky=NE)

        GUI.date_label = Label(time_frame, text=strftime("%A, %B %d", time.localtime()), fg='white', bg='black',
                               font=self.normalFont)
        GUI.date_label.grid(row=1, column=0, sticky=NE)

        # Frame for calendar info
        calendar_frame = Frame(self, width=400, height=500, bg='black')
        calendar_frame.grid(row=1, column=2, sticky=NE)
        GUI.calendar_label0 = Label(calendar_frame, text='\nUpcoming events:', fg='white', bg='black',
                                    font=self.mediumFont)
        GUI.calendar_label0.grid(row=0, column=0, sticky=NE)
        
        

        self.configure(background='black')

    def updateGUI(self):
        # Constantly updates the time until the program is stopped
        GUI.time_label.configure(text=strftime("%I:%M %p", time.localtime()))
        GUI.date_label.configure(text=strftime("%A, %B %d", time.localtime()))

        window.after(1000, mirror.updateGUI)


   
        

       

def close_escape(event=None):
    print('Smart mirror closed')
    window.destroy()


window = Tk()
window.title("Smart Mirror")
window.geometry('1920x1080')
window.configure(background='black')

#Removes borders from GUI and implements quit via esc
window.overrideredirect(1)
window.overrideredirect(0)
window.attributes("-fullscreen", True)
window.wm_attributes("-topmost", 1)
window.focus_set()

window.bind("<Escape>", close_escape)

mirror = GUI(window)
mirror.setupGUI()
window.after(1000, mirror.updateGUI)

window.mainloop()
