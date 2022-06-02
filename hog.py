import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import skimage.feature as ft

img = cv2.imread('1_1.jpg')

# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (0,0), fx=0.1, fy=0.1)[:144,:144]
print(img.shape)
# get hog compute
out, hog_img = ft.hog(img, orientations=36, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=True, multichannel=True, block_norm='L1', feature_vector=False)
# out = out.reshape(18, 18, -1)
print(out.shape)
print(hog_img.shape)
# show
plt.imshow(hog_img, cmap='gray')
plt.show()
