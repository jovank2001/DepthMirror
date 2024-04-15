"""
File: getImages.py
Author: Jovan Koledin
Email: jovank2001@gmail.com
Github: https://github.com/jovank2001/DepthMirror
Description: Script for saving pictures from IMX219-83 Stereo Camera Module for calibration
Usage: Run and click 'enter' key to save pictures from both cams
Created: April 5, 2024
Version: 1.0
License: MIT License
"""
from picamera2 import Picamera2, Preview
import time
import cv2 as cv
from pprint import *

#Settings
resolution = (640, 480) #(Width, Height)
fps = 20
wait = (1/fps)*1000 #Wait in milliseconds between frames

#Create camera objects, ensure 'unpacked' data format and set resolution
camR = Picamera2(0)
camL = Picamera2(1)
configR = camR.create_preview_configuration(raw={'format':'SRGGB8','size': resolution})
configL = camL.create_preview_configuration(raw={'format':'SRGGB8','size': resolution})
camR.configure(configR)
camL.configure(configL)

#Display settings
pprint(camR.sensor_modes)
print(camR.preview_configuration.raw) #Verify settings
print(camL.preview_configuration.raw) #Verify settings

#Capture images after key press numPics times
camR.start(show_preview = True)
camL.start(show_preview = True)
numPics = 5
imageCount = 0

#Get the images every keypress
while imageCount < numPics:

    input("Click Enter to capture pics")
    fPathR = "/calibrate/images/rightCam/imgR"+str(imageCount)+".jpg"
    fPathL = "/calibrate/images/leftCam/imgL"+str(imageCount)+".jpg"
    camR.capture_file(fPathR)
    camL.capture_file(fPathL)
    print("Images captured")
    imageCount += 1









    
    



