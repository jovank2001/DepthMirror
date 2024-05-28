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
import threading
import queue


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

def depth(imgL, imgR, data, focalLength, baselineLength):

    imgL = (imgL * .8).astype(np.uint8)
    imgR = (imgR * .8).astype(np.uint8)
    rectL, rectR = rectifyImages(imgL, imgR, data)
    depth = computeDepthSGBM(rectL, rectR, focalLength, baselineLength)
    print(depth.shape)
    return depth

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
    #depth_norm = cv.normalize(depth, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    
    # Calculate depth map from disparity
    with np.errstate(divide='ignore', invalid='ignore'):
        depth = np.where(disparity > 0, (focalLength * baselineLength) / np.clip(disparity, 1, np.inf), 0)

    # Optional: Apply a Gaussian filter to smooth the depth map
    depth = cv.GaussianBlur(depth, (5, 5), 0)
    depth_norm = cv.normalize(depth, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    
    return depth

def initialize_cameras(resolution=(640, 480)):
    """ Initializes and configures two camera instances. """
    camR = Picamera2(0)
    camL = Picamera2(1)
    config = {'format': 'SRGGB8', 'size': resolution}
    camR.configure(camR.create_preview_configuration(raw=config))
    camL.configure(camL.create_preview_configuration(raw=config))
    camR.start(show_preview=False)
    camL.start(show_preview=False)
    return camR, camL

def compute_fps(last_time):
    """ Computes the current frames per second. """
    current_time = time.time()
    fps = 1 / (current_time - last_time) if last_time else 0
    return current_time, fps

def capture_frames(camR, camL, frame_queue):
    """ Captures frames from cameras and puts them into a queue. """
    while not stop_event.is_set():
        frameR = cv.cvtColor(camR.capture_array('main'), cv.COLOR_BGR2GRAY)
        frameL = cv.cvtColor(camL.capture_array('main'), cv.COLOR_BGR2GRAY)
        frame_queue.put((frameR, frameL))

def process_depth(frame_queue, data, focalLength, baselineLength):
    """ Processes frames from the queue to compute the depth map. """
    last_time = 0
    while not stop_event.is_set() or not frame_queue.empty():
        try:
            frameR, frameL = frame_queue.get(timeout=1)
            depth_map = depth(frameL, frameR, data, focalLength, baselineLength)
            last_time, fps = compute_fps(last_time)
            cv.putText(depth_map, f"FPS: {fps:.2f}", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
            cv.imshow("Depth Map", depth_map)
            if cv.waitKey(1) == ord('q'):
                stop_event.set()
        except queue.Empty:
            continue

def main():
    frame_queue = queue.Queue(maxsize=10)
    global stop_event
    stop_event = threading.Event()
    
    focalLength = 2.6  # mm
    baselineLength = 60  # Distance between cameras [mm]
    calibrationDataPath = "calibrate/cameraData/calibrationData.npz"
    data = np.load(calibrationDataPath, allow_pickle=True)

    camR, camL = initialize_cameras()
    #cv.namedWindow("Depth Map")

    # Create threads for capturing and processing
    capture_thread = threading.Thread(target=capture_frames, args=(camR, camL, frame_queue))
    process_thread = threading.Thread(target=process_depth, args=(frame_queue, data, focalLength, baselineLength))

    # Start threads
    capture_thread.start()
    process_thread.start()

    try:
        capture_thread.join()
        process_thread.join()
    finally:
        camR.stop()
        camL.stop()
        cv.destroyAllWindows()

if __name__ == '__main__':
    main()