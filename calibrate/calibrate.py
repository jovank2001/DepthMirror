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
import time

#Camera Specs
focalLength = 2.6 #mm
angleOfView = {"diagonal":83, 
               "horizontal":73, 
               "vertical":50} #Degrees
baselineLength = 60 #Distance between cameras [mm]
fps = 30 
res = (1000, 750)
chessDims = (7,7)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessDims[0]*chessDims[1],3), np.float32)
objp[:,:2] = np.mgrid[0:chessDims[0],0:chessDims[1]].T.reshape(-1,2)
 
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

for i in range(2):

    if i == 0:
        lr = 'left'
    else:
        lr = 'right'
    
    images = glob.glob('calibrate/images/'+lr+'Cam/*.jpg')
    
    for fname in images:
        img = cv.imread(fname)
        imgsm = cv.resize(img, res)
        gray = cv.cvtColor(imgsm, cv.COLOR_RGB2GRAY)

        cv.imshow("gray", gray)
        cv.waitKey(0)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(imgsm, chessDims,  None)
        print(ret)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            
            # Draw and display the corners
            cv.drawChessboardCorners(imgsm, chessDims, corners2, ret)
            cv.imshow('img:'+str(fname), imgsm)
            cv.waitKey(0)
        
    cv.destroyAllWindows()

    #Calculate camera matrix and distortion coefficient and save
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    np.save('calibrate/cameraData/'+lr+'Mtx', mtx)
    np.save('calibrate/cameraData/'+lr+'Dist', dist)


