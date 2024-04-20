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

        grayL = cv.cvtColor(imgL, cv.COLOR_RGB2GRAY)
        grayR = cv.cvtColor(imgR, cv.COLOR_RGB2GRAY)

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
 
            # Draw and display the corners            
            cv.drawChessboardCorners(imgL, chessDims, corners2L, retL)
            cv.imshow('imgL', imgL)
            cv.waitKey(0)

            cv.drawChessboardCorners(imgR, chessDims, corners2R, retR)
            cv.imshow('imgR', imgR)
            cv.waitKey(0)
            
        if not retL or not retR:
            print(f"Chessboard corners not detected in pair ({imgPathL}, {imgPathR}). Skipping...")
            continue  # Skip this pair of images

    # Calibrate each camera
    retL, mtxL, distL, rvecsL, tvecsL = cv.calibrateCamera(objPoints, imgPointsL, grayL.shape[::-1], None, None)
    retR, mtxR, distR, rvecsR, tvecsR = cv.calibrateCamera(objPoints, imgPointsR, grayR.shape[::-1], None, None)

    # Stereo calibration
    stereoCalibrateRetval, mtxL, distL, mtxR, distR, R, T, E, F = cv.stereoCalibrate(
        objPoints, imgPointsL, imgPointsR, mtxL, distL,
        mtxR, distR, grayL.shape[::-1],
        criteria=criteria, flags=cv.CALIB_FIX_INTRINSIC
    )

    # After calibration, calculate re-projection errors to assess the quality
    mean_errorL = 0
    for i in range(len(objPoints)):
        imgpoints2, _ = cv.projectPoints(objPoints[i], rvecsL[i], tvecsL[i], mtxL, distL)
        error = cv.norm(imgPointsL[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_errorL += error
    mean_errorL /= len(objPoints)
    print(f"Mean Re-projection Error, Left Camera: {mean_errorL}")

    mean_errorR = 0
    for i in range(len(objPoints)):
        imgpoints2, _ = cv.projectPoints(objPoints[i], rvecsR[i], tvecsR[i], mtxR, distR)
        error = cv.norm(imgPointsR[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_errorR += error
    mean_errorR /= len(objPoints)
    print(f"Mean Re-projection Error, Right Camera: {mean_errorR}")

    # Stereo Rectification
    R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(
        mtxL, distL,
        mtxR, distR,
        grayL.shape[::-1], R, T,
        flags=cv.CALIB_ZERO_DISPARITY, alpha=0
    )

    # Save the calibration and rectification data
    np.savez(saveDataPath, mtxL=mtxL, distL=distL,
             mtxR=mtxR, distR=distR,
             R1=R1, R2=R2, P1=P1, P2=P2, Q=Q, roi1=roi1, roi2=roi2, F=F)

    print("Calibration and rectification data saved to", saveDataPath)

def drawEpipolarLines(img1, img2, F):
    '''
    Draw epipolar lines for feature points across both stereo images using the fundamental matrix F,
    ensuring that the lines are drawn on both images for each point in both images.
    Ex:
    imgLEpilines, imgREpilines = calibrate.drawEpipolarLines(rectL, rectR, data['F'])
    # Display the images with epipolar lines
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(cv.cvtColor(imgLEpilines, cv.COLOR_BGR2RGB))
    plt.title('Left Image with Epipolar Lines')
    plt.subplot(122)
    plt.imshow(cv.cvtColor(imgREpilines, cv.COLOR_BGR2RGB))
    plt.title('Right Image with Epipolar Lines')
    plt.show()
    '''
    # Convert images to grayscale for feature detection
    if len(img1.shape) == 3:
        img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    else:
        img1_gray = img1

    if len(img2.shape) == 3:
        img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    else:
        img2_gray = img2

    # Initialize ORB detector
    orb = cv.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1_gray, None)
    kp2, des2 = orb.detectAndCompute(img2_gray, None)

    # Create a BFMatcher object with distance measurement and match descriptors
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches by their distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract the point positions from the good matches
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Calculate epilines corresponding to the points in each image
    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F).reshape(-1, 3)

    # Create copies of the original images to draw lines on
    img1_color = cv.cvtColor(img1_gray, cv.COLOR_GRAY2BGR)
    img2_color = cv.cvtColor(img2_gray, cv.COLOR_GRAY2BGR)

    # Draw the epilines on both images
    for r, pt1, pt2 in zip(lines1, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        
        # Correctly format and typecast the point coordinates
        pt1 = (int(pt1[0][0]), int(pt1[0][1]))
        pt2 = (int(pt2[0][0]), int(pt2[0][1]))

        # Draw line from points in img2 on both img1 and img2
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [img1_color.shape[1], -(r[2]+r[0]*img1_color.shape[1])/r[1]])
        img1_color = cv.line(img1_color, (x0, y0), (x1, y1), color, 1)
        img2_color = cv.line(img2_color, (x0, y0), (x1, y1), color, 1)
        img2_color = cv.circle(img2_color, (pt2[0], pt2[1]), 5, color, -1)

    for r, pt1, pt2 in zip(lines2, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        
        # Correctly format and typecast the point coordinates
        pt1 = (int(pt1[0][0]), int(pt1[0][1]))
        pt2 = (int(pt2[0][0]), int(pt2[0][1]))

        # Draw line from points in img1 on both img1 and img2
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [img2_color.shape[1], -(r[2]+r[0]*img2_color.shape[1])/r[1]])
        img1_color = cv.line(img1_color, (x0, y0), (x1, y1), color, 1)
        img2_color = cv.line(img2_color, (x0, y0), (x1, y1), color, 1)
        img1_color = cv.circle(img1_color, (pt1[0], pt1[1]), 5, color, -1)

    return img1_color, img2_color
