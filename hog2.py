from cv2 import HOGDescriptor
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

img = cv2.imread('1_1.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[:144,:144]
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[:296,:296]
# img = cv2.resize(img, (0,0), fx=0.1, fy=0.1)

hog_descriptor = cv2.HOGDescriptor(
    _winSize=(144, 144),
    _blockSize=(16, 16),
    _blockStride=(8, 8),
    _cellSize=(8, 8),
    _nbins=36,
    _signedGradient=True
)
hog = hog_descriptor.compute(img)
hog = hog.reshape(17, 17, -1)
# hog_visual = hog_descriptor.computeGradient(img)
# print(hog_visual.shape)
print(hog.shape)