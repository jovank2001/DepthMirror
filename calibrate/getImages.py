"""
File: getImages.py
Author: Jovan Koledin
Email: jovank2001@gmail.com
Github: https://github.com/jovank2001/DepthMirror
Description: Script for saving pictures from IMX219-83 Stereo Camera Module for calibration
Usage: Run and click 's' to save pictures with chefrom both cameras and 'q' to quit.
Created: April 5, 2024
Version: 1.0
License: MIT License
"""
from picamera2 import Picamera2, Preview
import time
import cv2 as cv

#Settings
resolution = (640, 480) #(Width, Height)
fps = 20
wait = (1/fps)*1000 #Wait in milliseconds between frames

#Create camera objects, ensure 'unpacked' data format and set resolution
camR = Picamera2(0)
camL = Picamera2(1)
config = camR.create_preview_configuration(raw={'format': 'SBGGR8', 'size': resolution})
camR.configure(config)
camL.configure(config)
print(camR.preview_configuration.main) #Verify settings
print(camL.preview_configuration.main) #Verify settings

#Capture images after key press numPics times
camR.start()
camL.start()
numPics = 5
imageCount = 0

while imageCount < numPics:

    cv.waitKey(0)
    fPathR = "calibrate/images/rightCam/imgR"+str(imageCount)+".jpg"
    fPathL = "calibrate/images/leftCam/imgL"+str(imageCount)+".jpg"
    camR.capture_file(fPathR)
    camL.capture_file(fPathL)
    print("Images captured")
    imageCount += 1









    
    



