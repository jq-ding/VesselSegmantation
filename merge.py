import cv2
import numpy as np
import imageio
from skimage import img_as_ubyte
from skimage import io,data
from skimage import morphology,data
import matplotlib.pyplot as plt


def delete_circle(img):

    rowNum = len(img)     #gao and hang
    colNum = len(img[0])  #kuan and lie

    for i in range(rowNum):
        for j in range(colNum):
            if img[i][j] == 255:
                img[i][j] = 0
    for i in range(rowNum):
        for j in range(colNum):
            img[i][j] *=255
    return img


# 直接变255
def fusion(raw_img, thick, thin):
    rowNum = len(thick)     #gao and hang
    colNum = len(thick[0])  #kuan and lie

    newImg = np.zeros((rowNum, colNum), np.uint8)

    for i in range(rowNum):
        for j in range(colNum):
            if thick[i][j] > 0.45 or thin[i][j] > 0.45 or raw_img[i][j] > 0.45:
                newImg[i][j] = 255
    return newImg

# 三个之中某个超过阈值变255
def fusion2(raw_img, thick, thin):
    rowNum = len(thick)     #gao and hang
    colNum = len(thick[0])  #kuan and lie

    newImg = np.zeros((rowNum, colNum), np.uint8)

    for i in range(rowNum):
        for j in range(colNum):
            if thick[i][j] > 0.45 or thin[i][j] > 0.45 or raw_img[i][j] > 0.45:
                newImg[i][j] = 255
    return newImg

# 三个取均值超过阈值
def fusion3(raw_img, thick, thin):
    rowNum = len(thick)     #gao and hang
    colNum = len(thick[0])  #kuan and lie

    newImg = np.zeros((rowNum, colNum), np.uint8)

    for i in range(rowNum):
        for j in range(colNum):
            avg = (thick[i][j] + thin[i][j] + raw_img[i][j]) / 3
            if avg > 0.2:
                newImg[i][j] = 255
    return newImg

# 三个有两个大于阈值
def fusion4(raw_img, thick, thin):
    rowNum = len(thick)     #gao and hang
    colNum = len(thick[0])  #kuan and lie

    newImg = np.zeros((rowNum, colNum), np.uint8)

    for i in range(rowNum):
        for j in range(colNum):
            if (thick[i][j] > 0.3 and thin[i][j] > 0.3) or (thick[i][j] > 0.3 and raw_img[i][j] > 0.3) or (thin[i][j] > 0.3 and raw_img[i][j] > 0.3):
                newImg[i][j] = 255
    return newImg

rawimg = imageio.mimread(r'C:\Users\DJQ\Desktop\results\for_test\first\01_manual1.gif')
img = plt.imread(r'C:/Users/DJQ\Desktop/results/for_test/first/testPrediction0.png')
mask = imageio.mimread(r'C:\Users\DJQ\Desktop\results\for_test\first\01_test_mask.gif')
# skt = imageio.mimread(r'C:\Users\DJQ\Downloads\DRIVE\test\1st_manual\01_manual1.gif')
# thick = imageio.mimread(r'C:/Users/DJQ/Downloads/Retina-Unet-master (1)/Retina-Unet-master/thin_only/thick.gif')
thick = plt.imread('C:/Users/DJQ/Desktop/thick_ce/testPrediction0.png')
thin = plt.imread('C:/Users/DJQ/Desktop/thin_ce/testPrediction0.png')

mask = mask[0]

fusion = fusion(img, thick, thin)

cv2.imshow('fusion', fusion)

io.imsave('fusion.png', fusion)

cv2.waitKey(0)
