"""
File: calibrateCams.py
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
res = (1000, 750)
chessDims = (7,7)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
ptsL, ptsR = [], []
imgSize = None

#Grab images
imgsL = list(sorted(glob.glob('./calibrate/images/leftCam/imgLW*.png')))
imgsR = list(sorted(glob.glob('./calibrate/images/rightCam/imgRW*.png')))
assert len(imgsL) == len(imgsR)

#Calibrate chessboard images
for imgPathL, imgPathR in zip(imgsL, imgsR):
    imgL = cv.imread(imgL, cv.IMREAD_GRAYSCALE)
    imgR = cv.imread(imgR, cv.IMREAD_GRAYSCALE)
    if imgSize is None:
        imgSize = (imgL.shape[1], imgL.shape[0])
    
    resL, cornersL = cv.findChessboardCorners(imgL, chessDims)
    resR, cornersR = cv.findChessboardCorners(imgR, chessDims)
    
    cornersL = cv.cornerSubPix(imgL, cornersL, (10, 10), (-1,-1),
                                    criteria)
    cornersR = cv.cornerSubPix(imgR, cornersR, (10, 10), (-1,-1), 
                                     criteria)
    
    ptsL.append(cornersL)
    ptsR.append(cornersR)

pattern_points = np.zeros((np.prod(chessDims), 3), np.float32)
pattern_points[:, :2] = np.indices(chessDims).T.reshape(-1, 2)
pattern_points = [pattern_points] * len(imgsL)

err, Kl, Dl, Kr, Dr, R, T, E, F = cv.stereoCalibrate(
    pattern_points, ptsL, ptsR, None, None, None, None, imgSize, flags=0)