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
from matplotlib import pyplot as plt

#Camera Specs
focalLength = 2.6 #mm
angleOfView = {"diagonal":83, 
               "horizontal":73, 
               "vertical":50} #Degrees
baselineLength = 60 #Distance between cameras [mm]
fps = 30 
resolution = [640, 480] 

#Cameras
capR = cv.VideoCapture(0)
capL = cv.VideoCapture(1)

