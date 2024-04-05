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

import cv2 as cv

#Settings
resolution = [640, 480] #[Width, Height]
fps = 20
wait = (1/fps)*1000 #Wait in milliseconds between frames

#Create camera objects and set resolution
capR = cv.VideoCapture(0)
capL = cv.VideoCapture(1)
capR.set(cv.CAP_PROP_FRAME_WIDTH, resolution[0])
capR.set(cv.CAP_PROP_FRAME_HEIGHT, resolution[1])
capL.set(cv.CAP_PROP_FRAME_WIDTH, resolution[0])
capL.set(cv.CAP_PROP_FRAME_HEIGHT, resolution[1])

#Capture Images
num = 0 # Index for image capture loop below
while capR.isOpened() and capL.isOpened():

    successR, imgR = capR.read()
    successL, imgL = capL.read()

    #Wait 50ms before displaying next frame 
    k = cv.waitKey(wait) 

    if k == ord('q'): #Quit on 'q' keypress
        break
    elif k == ord('s'): #Take photo on 's' keypress
        cv.imwrite('calibrate/images/rightCam/capR' + str(num) + '.png', imgR)
        cv.imwrite('calibrate/images/leftCam/capL' + str(num) + '.png', imgL)
        print("Pics taken" + str(num))
        num += 1

    cv.imshow("Right", imgR)
    cv.imshow("Left", imgL)

capR.release()
capL.release()

cv.destroyAllWindows()


        





    
    



