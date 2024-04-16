"""
File: depth.py
Author: Jovan Koledin
Email: jovank2001@gmail.com
Github: https://github.com/jovank2001/DepthMirror
Description: Script for creating our stereo depth map from calibration parameters.
Created: April 5, 2024
Version: 1.0
License: MIT License
"""

import cv2 as cv
import numpy as np
import calibrate
#import pics

def loadTestImages(imagePathL, imagePathR):
    # Load stereo images
    imgL = cv.imread(imagePathL, cv.IMREAD_GRAYSCALE)
    imgR = cv.imread(imagePathR, cv.IMREAD_GRAYSCALE)
    return imgL, imgR

def rectifyImages(imgL, imgR, params):
    # Unpack parameters
    R1, R2, P1, P2, Q = params['R1'], params['R2'], params['P1'], params['P2'], params['Q']

    # Compute the rectification transformation maps
    map1L, map2L = cv.initUndistortRectifyMap(
        params['mtxL'], params['distL'], R1, P1, imgL.shape, cv.CV_16SC2)
    map1R, map2R = cv.initUndistortRectifyMap(
        params['mtxR'], params['distR'], R2, P2, imgR.shape, cv.CV_16SC2)

    # Remap the images using the rectification maps
    rectL = cv.remap(imgL, map1L, map2L, cv.INTER_LINEAR)
    rectR = cv.remap(imgR, map1R, map2R, cv.INTER_LINEAR)

    return rectL, rectR

def computeDepth(rectL, rectR, focalLength, baselineLength):
    # Create StereoBM object
    stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(rectL, rectR)
    
    # Replace zero disparity with a small minimum value to avoid division by zero
    disparity[disparity == 0] = 0.1
    depth = np.zeros(disparity.shape, np.float32)
    
    # Calculate the depth
    depth = focalLength * baselineLength / disparity

    return depth

def main():
    #Camera Specs
    focalLength = 2.6 #mm
    angleOfView = {"diagonal":83, 
                    "horizontal":73, 
                    "vertical":50} #Degrees
    baselineLength = 60 #Distance between cameras [mm]
    chessDims = (9,6) #Inside corners of calibration chessboard
    caibrationDataPath = "calibrate/cameraData/calibrationData.npz" 
    calibrationImagesPathL = "calibrate/images/leftCam/imgL*.jpg"
    calibrationImagesPathR = "calibrate/images/rightCam/imgR*.jpg"
    testImagePathL = "calibrate/images/leftCam/testL.jpg"
    testImagePathR = "calibrate/images/rightCam/testR.jpg"

    #Take pictures of the chessboard for calibration data generation
    #pics.takeCalibrationPics()

    #Generate calibration data
    calibrate.generateCalibrationData(
        calibrationImagesPathL, calibrationImagesPathR, chessDims, caibrationDataPath)

    # Load saved calibration and rectification data
    data = np.load(caibrationDataPath)
    
    # Load stereo images
    imgL, imgR = loadTestImages(testImagePathL, testImagePathR)
    
    # Rectify images
    rectL, rectR = rectifyImages(imgL, imgR, data)
    
    # Compute the disparity map
    depth = computeDepth(rectL, rectR, focalLength, baselineLength)

    # Display the disparity map
    cv.imshow('Depth Map', depth)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
