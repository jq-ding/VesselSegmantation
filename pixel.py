import cv2
import numpy as np
import imageio
from skimage import io,data
import matplotlib.pyplot as plt
import os


def delete_circle(img, mask):

    rowNum = len(img)     #gao and hang
    colNum = len(img[0])  #kuan and lie

    for i in range(rowNum):
        for j in range(colNum):
            if mask[i][j] == 0:
                img[i][j] = 0
    return img


def calDSI(binary_GT, binary_R):
    row, col = binary_GT.shape  # 矩阵的行与列
    DSI_s, DSI_t = 0, 0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255 and binary_R[i][j] == 255:
                DSI_s += 1
            if binary_GT[i][j] == 255:
                DSI_t += 1
            if binary_R[i][j] == 255:
                DSI_t += 1
    DSI = 2*DSI_s/DSI_t    # print(DSI)
    return DSI


def calPrecision(binary_GT, binary_R):
    row, col = binary_GT.shape  # 矩阵的行与列
    P_s, P_t = 0, 0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255 and binary_R[i][j] == 1:
                P_s += 1
            if binary_R[i][j] == 1:
                P_t += 1
    Precision = P_s/P_t
    return Precision


def calRecall(binary_GT, binary_R):
    row, col = binary_GT.shape  # 矩阵的行与列
    R_s, R_t = 0, 0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255 and binary_R[i][j] == 1:
                R_s += 1
            if binary_GT[i][j] == 255:
                R_t += 1
    Recall = R_s/R_t
    return Recall


def separatePixel(binary_GT, binary_R, mask):
    row, col = binary_GT.shape  # 矩阵的行与列
    TP, TN, FP, FN, total, pos = 0, 0, 0, 0, 0, 0
    for i in range(row):
        for j in range(col):
            if mask[i][j] == 255:
                if binary_GT[i][j] == 255 and binary_R[i][j] == 255:
                    TP += 1
                if binary_GT[i][j] == 255 and binary_R[i][j] == 0:
                    FN += 1
                if binary_GT[i][j] == 0 and binary_R[i][j] == 255:
                    FP += 1
                if binary_GT[i][j] == 0 and binary_R[i][j] == 0:
                    TN += 1
                if binary_GT[i][j] == 255 or binary_GT[i][j] == 0:
                    total += 1
                if binary_GT[i][j] == 255:
                    pos += 1
                if binary_R[i][j] == 255:
                    pos += 1

    accuracy = (TP + TN) / total
    # precision = TP / (TP + FP)
    se = TP / (TP + FN)  # se tpr
    # FPR = FP / (FP + TN)
    dice = 2 * TP / pos
    sp = TN / (FP + TN)
    # F1 = 2 * precision * recall / (precision + recall)
    return se, sp, dice, accuracy


pre = plt.imread('pre20.png')

gt = io.imread('20_thinmanual1.gif')
mask = plt.imread(r'DRIVE/test/mask/20_test_mask.gif')

index1 = separatePixel(gt, pre*255, mask)
print(index1)

