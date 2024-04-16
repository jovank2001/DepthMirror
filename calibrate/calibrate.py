"""
File: calibrate.py
Author: Jovan Koledin
Email: jovank2001@gmail.com
Github: https://github.com/jovank2001/DepthMirror
Description: Script for generating calibration params for the IMX219-83 Stereo Camera Module
Usage: Run after you have images taken from both cameras of the chess board (pics.py)
Created: April 5, 2024
Version: 1.0
License: MIT License
"""

import cv2 as cv
import numpy as np
import glob

def generateCalibrationData(imagesPathL, imagesPathR, chessDims, saveDataPath):
    # Termination criteria for corner refinement
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Object points array, with zero coordinates for z since the checkerboard is flat
    objp = np.zeros((chessDims[0] * chessDims[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessDims[0], 0:chessDims[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points for both cameras
    objPoints = []  # 3D points in real world space
    imgPointsL = []  # 2D points in image plane from left camera
    imgPointsR = []  # 2D points in image plane from right camera

    # Collect images from both directories
    imagesL = sorted(glob.glob(imagesPathL))
    imagesR = sorted(glob.glob(imagesPathR))

    assert len(imagesL) == len(imagesR), "Mismatch in number of left and right images."

    for imgPathL, imgPathR in zip(imagesL, imagesR):
        imgL = cv.imread(imgPathL)
        imgR = cv.imread(imgPathR)

        grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
        grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

        # Find the chess board corners in both images
        retL, cornersL = cv.findChessboardCorners(grayL, chessDims, None)
        retR, cornersR = cv.findChessboardCorners(grayR, chessDims, None)

        # If found in both images, add object points and image points
        if retL and retR:
            objPoints.append(objp)
            corners2L = cv.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
            corners2R = cv.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
            imgPointsL.append(corners2L)
            imgPointsR.append(corners2R)

    # Calibrate each camera
    retL, mtxL, distL, rvecsL, tvecsL = cv.calibrateCamera(objPoints, imgPointsL, grayL.shape[::-1], None, None)
    retR, mtxR, distR, rvecsR, tvecsR = cv.calibrateCamera(objPoints, imgPointsR, grayR.shape[::-1], None, None)

    # Stereo calibration
    stereoCalibrateRetval, mtxL, distL, mtxR, distR, R, T, E, F = cv.stereoCalibrate(
        objPoints, imgPointsL, imgPointsR, mtxL, distL,
        mtxR, distR, grayL.shape[::-1],
        criteria=criteria, flags=cv.CALIB_FIX_INTRINSIC
    )

    # Stereo Rectification
    R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(
        mtxL, distL,
        mtxR, distR,
        grayL.shape[::-1], R, T,
        flags=cv.CALIB_ZERO_DISPARITY, alpha=-1
    )

    # Save the calibration and rectification data
    np.savez(saveDataPath, mtxL=mtxL, distL=distL,
             mtxR=mtxR, distR=distR,
             R1=R1, R2=R2, P1=P1, P2=P2, Q=Q, roi1=roi1, roi2=roi2)

    print("Calibration and rectification data saved to", saveDataPath)
