"""
File: pics.py
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
import cv2 as cv
from pprint import *

        
#Settings
resolutionC = (1680, 1240) #(Width, Height)
resolutionT = (640, 480) #(Width, Height)
  
#Create camera objects, ensure 'unpacked' data format and set resolution
camR = Picamera2(0)
camL = Picamera2(1)
configR = camR.create_preview_configuration(raw={'format':'SRGGB8', 'size':resolutionT})
configL = camL.create_preview_configuration(raw={'format':'SRGGB8', 'size':resolutionT})
camR.configure(configR)
camL.configure(configL)

#Display set
print(camR.camera_configuration()) #Verify settings
print(camL.camera_configuration()) #Verify settings

#Capture images after key press numPics times
camL.start(show_preview = False)
camR.start(show_preview = False)
numPics = 15
imageCount = 0

#Get the images every keypress
while imageCount < numPics:

    input("Click Enter to capture pics")
    fPathR = "images/right/imgR"+str(imageCount)+".jpg"
    fPathL = "images/left/imgL"+str(imageCount)+".jpg"
    camR.capture_file(fPathR)
    camL.capture_file(fPathL)
    imgR = cv.imread(fPathR)
    imgL = cv.imread(fPathL)
    cv.imshow("Right", imgR)
    cv.imshow("Left", imgL)
    cv.waitKey(0)
    print("Images captured")
    imageCount += 1









        
        



