import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
import random

N = 3
S = N + 1
O = 5
k = 2 ** (1/N)
sigma0 = 1.52

sigmas = [[(k**s)*sigma0*(1<<o) for s in range(S)] for o in range(O)]

ori_img = cv2.imread('2_1.jpg')
ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)

# ideal
pyr1 = []
for i in range(O):
    pyr1.append([])
    for j in range(S):
        _sigma = sigmas[i][j]
        _img = cv2.GaussianBlur(ori_img, (0,0), _sigma)
        stride = 2**i
        _img = _img[::stride, ::stride]
        pyr1[i].append(_img)

# downsample first
pyr2 = []
for i in range(O):
    pyr2.append([])
    stride = 2**i
    s_img = ori_img[::stride, ::stride]
    for j in range(S):
        _sigma = sigmas[i][j]
        _img = cv2.GaussianBlur(s_img, (0,0), _sigma)
        pyr2[i].append(_img)

# iterate1
pyr3 = []
for i in range(O):
    pyr3.append([])
    for j in range(S):
        if i == 0 and j == 0:
            _sigma = sigmas[i][j]
            _img = cv2.GaussianBlur(ori_img, (0,0), _sigma)
            pyr3[i].append(_img)
        elif j == 0:
            _img = pyr3[i-1][-1]
            _img = _img[::2, ::2]
            pyr3[i].append(_img)
        else:
            _sigma = math.sqrt(sigmas[i][j]**2 - sigmas[i][0]**2)
            _img = cv2.GaussianBlur(pyr3[i][0], (0,0), _sigma)
            pyr3[i].append(_img)

# iterate2
pyr4 = []
for i in range(O):
    pyr4.append([])
    for j in range(S):
        if i == 0 and j == 0:
            _sigma = sigmas[i][j]
            _img = cv2.GaussianBlur(ori_img, (0,0), _sigma)
            pyr4[i].append(_img)
        elif j == 0:
            _img = pyr4[i-1][-1]
            _img = _img[::2, ::2]
            pyr4[i].append(_img)
        else:
            _sigma = math.sqrt(sigmas[i][j]**2 - sigmas[i][j-1]**2)
            _img = cv2.GaussianBlur(pyr4[i][j-1], (0,0), _sigma)
            pyr4[i].append(_img)

# iterate3 (opencv)
pyr5 = []
for i in range(O):
    pyr5.append([])
    for j in range(S):
        if i == 0 and j == 0:
            _sigma = sigmas[i][j]
            _img = cv2.GaussianBlur(ori_img, (0,0), _sigma)
            pyr5[i].append(_img)
        elif j == 0:
            _img = pyr5[i-1][-1]
            _img = _img[::2, ::2]
            pyr5[i].append(_img)
        else:
            _sigma = math.sqrt(sigmas[0][j]**2 - sigmas[0][j-1]**2)
            _img = cv2.GaussianBlur(pyr5[i][j-1], (0,0), _sigma)
            pyr5[i].append(_img)


def difference(p1, p2, name):
    diff = 0
    n_pix = 0
    for i in range(O):
        for j in range(S):
            diff += np.sum(np.abs(p1[i][j] - p2[i][j]))
            n_pix += p1[i][j].size
    print(f'{name}:', diff / n_pix)

difference(pyr1, pyr2, 'downsample first')
difference(pyr1, pyr3, 'iterate1')
difference(pyr1, pyr4, 'iterate2')
difference(pyr1, pyr5, 'iterate3 (opencv)')