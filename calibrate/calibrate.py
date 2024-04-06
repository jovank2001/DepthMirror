"""
File: calibrate.py
Author: Jovan Koledin
Email: jovank2001@gmail.com
Github: https://github.com/jovank2001/DepthMirror
Description: Script for calibrating the IMX219-83 Stereo Camera Module 
Usage: Run after you have images taken from both cameras of the chess board
Created: April 5, 2024
Version: 1.0
License: MIT License
"""

import cv2 as cv
import numpy as np
import glob

#Camera Specs
focalLength = 2.6 #mm
angleOfView = {"diagonal":83, 
               "horizontal":73, 
               "vertical":50} #Degrees
baselineLength = 60 #Distance between cameras [mm]
fps = 30 
resolution = [3280, 2464] 
chessDims = (7,7)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessDims[0]*chessDims[1],3), np.float32)
objp[:,:2] = np.mgrid[0:chessDims[0],0:chessDims[1]].T.reshape(-1,2)
 
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
 
images = glob.glob('calibrate/images/leftCam/*.jpg')
 
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, chessDims, None)
    
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        
        # Draw and display the corners
        cv.drawChessboardCorners(img, chessDims, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)
    
cv.destroyAllWindows()
