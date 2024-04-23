"""
File: depthmirror.py
Author: Jovan Koledin
Email: jovank2001@gmail.com
Github: https://github.com/jovank2001/DepthMirror
Description: Creates our stereo depth map
Created: April 5, 2024
Version: 1.0
License: MIT License
"""

from picamera2 import Picamera2, Preview
import time
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
#import pics

def loadTestImages(imagePathL, imagePathR):
    # Load uncalibrated stereo images
    imgL = cv.imread(imagePathL, cv.IMREAD_GRAYSCALE)
    imgR = cv.imread(imagePathR, cv.IMREAD_GRAYSCALE)
    return imgL, imgR

def rectifyImages(imgL, imgR, params):
    # Unpack parameters
    R1, R2, P1, P2, Q = params['R1'], params['R2'], params['P1'], params['P2'], params['Q']

    # Compute the rectification transformation maps
    map1L, map2L = cv.initUndistortRectifyMap(
        params['mtxL'], params['distL'], R1, P1, imgL.shape[::-1], cv.CV_16SC2)
    map1R, map2R = cv.initUndistortRectifyMap(
        params['mtxR'], params['distR'], R2, P2, imgR.shape[::-1], cv.CV_16SC2)

    # Remap the images using the rectification maps
    rectL = cv.remap(imgL, map1L, map2L, cv.INTER_LINEAR)
    rectR = cv.remap(imgR, map1R, map2R, cv.INTER_LINEAR)

    return rectL, rectR

def depthKitti():
    '''
    To run pre trained Kitti Stereo Keras model:
    python3 inferencing.py \
    --left_dir /path/to/left/rectified_images \
    --right_dir /path/to/right/rectified_images \
    --num_adapt 0 \
    --weights_path /path/to/pretrained/model \
    --height 480 \
    --width 640 \
    --batch_size 1 
    '''
    pass

def depth(imgL, imgR, data, focalLength, baselineLength):

    imgL = np.clip(imgL * .8, 0, 255).astype(np.uint8)
    imgR = np.clip(imgR * .8, 0, 255).astype(np.uint8)
    rectL, rectR = rectifyImages(imgL, imgR, data)
    depth = computeDepthSGBM(rectL, rectR, focalLength, baselineLength)

    return depth


def computeDepthBM(rectL, rectR, focalLength, baselineLength):
    # Configure the Stereo Block Matching (SBM) algorithm
    numDisparities = 32  # Increased for better depth precision
    blockSize = 11       # Slightly reduced to balance detail against noise (must be odd)

    # Create StereoBM object with optimized parameters
    stereo = cv.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)
    
    # Additional parameters to improve the disparity calculation
    stereo.setMinDisparity(0)  # Set to 0 if unsure about the minimum disparity
    stereo.setTextureThreshold(10)  # Lower this if too many areas are considered untextured
    stereo.setUniquenessRatio(15)  # Lower value to increase potential matches
    stereo.setSpeckleWindowSize(100)  # Increased for better speckle filtering
    stereo.setSpeckleRange(32)  # Increased for better smoothness in disparity

    # Pre-filtering for better matching (especially in low-texture areas)
    stereo.setPreFilterType(cv.STEREO_BM_PREFILTER_XSOBEL)
    stereo.setPreFilterSize(9)
    stereo.setPreFilterCap(31)

    # Compute disparity map
    disparity = stereo.compute(rectL, rectR).astype(np.float32)

    # Normalization of the disparity map (optional, for visualization)
    #disp_norm = cv.normalize(disparity, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    
    # Calculate depth map from disparity
    # Use clip to handle negative and extreme values, avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        depth = np.where(disparity > 0, focalLength * baselineLength / np.clip(disparity, 1, np.inf), 0)

    # Apply a Gaussian filter to smooth the depth map, might be better than median for preserving edges
    depth = cv.GaussianBlur(depth, (5, 5), 0)

    return depth

def fill_missing_disparities(disparity_map, method='telea', radius=3):
    """
    Interpolates and fills missing disparities in a disparity map.
    
    Args:
        disparity_map (np.array): The input disparity map with missing values.
        method (str): Interpolation method, options are 'telea' or 'ns'.
        radius (int): The radius used for the inpainting algorithm.
    
    Returns:
        np.array: The disparity map with missing disparities filled.
    """
    # Identify missing disparities; this might need adjustment depending on how your data represents it
    missing_mask = (disparity_map <= 0).astype(np.uint8)
    
    # Choose the inpainting method
    if method == 'telea':
        inpaint_method = cv.INPAINT_TELEA
    else:
        inpaint_method = cv.INPAINT_NS
    
    # Perform inpainting to fill missing values
    filled_disparity = cv.inpaint(disparity_map, missing_mask, radius, inpaint_method)
    
    return filled_disparity

def computeDepthSGBM(rectL, rectR, focalLength, baselineLength):
    # Configure the Stereo Semi-Global Block Matching (SGBM) algorithm
    window_size = 5
    min_disp = 0
    num_disp = 16*2  # Number of disparity increments
    blockSize = 9  # Block size to match

    # Create StereoSGBM object
    stereo = cv.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=blockSize,
        P1=8 * 1 * window_size**2,  # Parameters controlling the disparity smoothness.
        P2=32 * 1 * window_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # Compute disparity map
    disparity = stereo.compute(rectL, rectR).astype(np.float32)
    disparity = (disparity / 16.0) - min_disp / 16

    # Normalize the disparity map for visualization (optional)
    disp_norm = cv.normalize(disparity, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    
    # Calculate depth map from disparity
    with np.errstate(divide='ignore', invalid='ignore'):
        depth = np.where(disp_norm > 0, (focalLength * baselineLength) / np.clip(disparity, 1, np.inf), 0)

    # Optional: Apply a Gaussian filter to smooth the depth map
    depth = cv.GaussianBlur(depth, (5, 5), 0)

    return depth

def main():
    #Camera Specs
    focalLength = 2.6 #mm
    angleOfView = {"diagonal":83, 
                    "horizontal":73, 
                    "vertical":50} #Degrees
    baselineLength = 60 #Distance between cameras [mm]
    chessDims = (9,6) #Inside corners of calibration chessboard
    calibrationDataPath = "calibrate/cameraData/calibrationData.npz" 
    calibrationImagesPathL = "calibrate/images/leftCam/imgL*.jpg"
    calibrationImagesPathR = "calibrate/images/rightCam/imgR*.jpg"
    resolutionT = (640, 480) #(Width, Height)

    #Take pictures of the chessboard for calibration data generation
    pics.takePics()

    #Generate calibration data
    #calibrate.generateCalibrationData(calibrationImagesPathL, calibrationImagesPathR, chessDims, calibrationDataPath)
    
    # Load saved calibration and rectification data
    data = np.load(calibrationDataPath)
    
    #Create camera objects, ensure 'unpacked' data format and set resolution
    camR = Picamera2(0)
    camL = Picamera2(1)
    configR = camR.create_preview_configuration(raw={'format':'SRGGB8', 'size':resolutionT})
    configL = camL.create_preview_configuration(raw={'format':'SRGGB8', 'size':resolutionT})
    camR.configure(configR)
    camL.configure(configL)

    #Display settings
    print(camR.camera_configuration()) #Verify settings
    print(camL.camera_configuration()) #Verify settings

    #Capture images after key press numPics times
    camL.start(show_preview = True)
    camR.start(show_preview = True)

    #Display Depth map
    frameCount = 0
    while True:
        frameR = camR.capture_array('main')
        frameL = camL.capture_array('main')
        depth = depth(frameL, frameR, data, focalLength, baselineLength)
        cv.imshow("Depth frame:" + frameCount, depth)
        if cv.waitKey(1) == ord('q'):  # press 'q' to quit
                break
        
if __name__ == '__main__':
    main()
