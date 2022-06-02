import cv2
import numpy as np

# gauss_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
# gauss_kernel = gauss_kernel / 16

im = cv2.imread('1_1.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gaussian = cv2.GaussianBlur(im, ksize=(5, 5), sigmaX=1, sigmaY=1)
# show
cv2.imshow('im_gaussian', im_gaussian)
cv2.waitKey(0)
