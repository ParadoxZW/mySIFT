import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

def drawLines(info, img):
    # draw lines
    for i in range(info.shape[0]):
        rand_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.line(img,(int(info[i,0]),int(info[i,2])),(int(info[i,1]),int(info[i,3])),rand_color,2)
    if len(img.shape) == 2:
        plt.imshow(img.astype(np.uint8),cmap='gray')
    else:
        plt.imshow(img.astype(np.uint8))
    plt.axis('off')
    plt.show()

img1 = cv2.imread('1_1.jpg')
img2 = cv2.imread('1_2.jpg')
img = np.hstack((img1,img2))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
info = np.load('info.npy')
drawLines(info, img)
