import cv2
import numpy as np
import imageio
from skimage import img_as_ubyte
from skimage import io,data
from skimage import morphology,data
import matplotlib.pyplot as plt

def getOutlineByMatrix(rawImg):
    rowNum = len(rawImg)     #gao and hang
    colNum = len(rawImg[0])  #kuan and lie

    newImg = np.ones((rowNum, colNum), np.uint8)

    # print(rowNum,'--',colNum)
    for i in range(rowNum):
        for j in range(colNum):
            if rawImg[i][j] > 0:
                top = i-1
                bot = i+1
                lef = j-1
                rig = j+1
                if i == 0:
                    top = 0
                if i == rowNum-1:
                    bot = rowNum-1
                if j == 0:
                    lef = 0
                if j == colNum-1:
                    rig = colNum-1
                # print(i,j,top,bot,lef,rig)
                if rawImg[top][j] ==0 or rawImg[bot][j] ==0 or rawImg[i][lef] ==0 or rawImg[i][rig] ==0:
                    newImg[i][j] = 255
                else:
                    newImg[i][j] = 0;
                    # print(rawImg[top][j],rawImg[bot][j],rawImg[i][lef],rawImg[i][rig])
            else:
                newImg[i][j] = 0   #  duoyu

    return  newImg


def deleteByOutline(rawImg, Outline):
    rowNum = len(rawImg)
    colNum = len(rawImg[0])

    newImg = np.ones((rowNum, colNum), np.uint8)
    deleteNum = 0

    for i in range(rowNum):
        for j in range(colNum):
            # newImg[i][j] = rawImg[i][j]
            newImg[i][j] = rawImg[i][j]
            if Outline[i][j] > 0:
                outlineNum = 0
                outlineNum2 = 0; # direct
                changeNum = 0;
                flag0 = 0;
                flag1 = 0;
                flag2 = 0;

                flag0 = getValueByIndex(i - 1, j - 1, Outline)
                outlineNum = outlineNum + flag0

                flag1 = getValueByIndex(i - 1, j, Outline)
                outlineNum2 = outlineNum2 + flag1
                if flag0 != flag1:
                    changeNum = changeNum + 1

                flag2 = getValueByIndex(i - 1, j + 1, Outline)
                outlineNum = outlineNum + flag2
                if flag1 != flag2:
                    changeNum = changeNum + 1
                flag1 = flag2

                flag2 = getValueByIndex(i, j + 1, Outline)
                outlineNum2 = outlineNum2 + flag2
                if flag1 != flag2:
                    changeNum = changeNum + 1
                flag1 = flag2

                flag2 = getValueByIndex(i + 1, j + 1, Outline)
                outlineNum = outlineNum + flag2
                if flag1 != flag2:
                    changeNum = changeNum + 1
                flag1 = flag2

                flag2 = getValueByIndex(i + 1, j, Outline)
                outlineNum2 = outlineNum2 + flag2
                if flag1 != flag2:
                    changeNum = changeNum + 1
                flag1 = flag2

                flag2 = getValueByIndex(i + 1, j - 1, Outline)
                outlineNum = outlineNum + flag2
                if flag1 != flag2:
                    changeNum = changeNum + 1
                flag1 = flag2

                flag2 = getValueByIndex(i, j - 1, Outline)
                outlineNum2 = outlineNum2 + flag2
                if flag1 != flag2:
                    changeNum = changeNum + 1

                if flag0 != flag2:
                    changeNum = changeNum + 1

                outlineNum = outlineNum2 + outlineNum

                noZeroNum = 0
                noZeroNum2 = 0
                noZeroNum = noZeroNum + getValueByIndex(i - 1, j, rawImg)
                noZeroNum = noZeroNum + getValueByIndex(i + 1, j, rawImg)
                noZeroNum = noZeroNum + getValueByIndex(i, j + 1, rawImg)
                noZeroNum = noZeroNum + getValueByIndex(i, j - 1, rawImg)

                noZeroNum2 = noZeroNum2 + getValueByIndex(i-1, j-1, rawImg)
                noZeroNum2 = noZeroNum2 + getValueByIndex(i+1, j+1, rawImg)
                noZeroNum2 = noZeroNum2 + getValueByIndex(i-1, j+1, rawImg)
                noZeroNum2 = noZeroNum2 + getValueByIndex(i+1, j-1, rawImg)
                noZeroNum2 = noZeroNum2 + noZeroNum

                if outlineNum == 2:
                    if outlineNum2 == 0 and noZeroNum > 0:
                        newImg[i][j] = 0  # delete
                        deleteNum = deleteNum + 1
                    if outlineNum2 == 1 and noZeroNum > 1:
                        newImg[i][j] = 0  # delete
                        deleteNum = deleteNum + 1
                    if outlineNum2 == 2 and noZeroNum2 > 2:
                        newImg[i][j] = 0  # delete
                        deleteNum = deleteNum + 1
                if outlineNum == 3:
                    if outlineNum2 == 1 or outlineNum2 == 2:
                        if changeNum == 4:
                            if noZeroNum2 > 3:
                                newImg[i][j] = 0  # delete
                                deleteNum = deleteNum + 1

    return newImg, deleteNum


def getValueByIndex(i,j,mtx):
    rowNum = len(mtx)
    colNum = len(mtx[0])
    res = 0;
    if i >= 0 and i < rowNum and j >= 0 and j < colNum:
        if mtx[i][j] > 0:
            res = 1
    return res


def getSkeletonByMatrix(rawImg):
    return 0


def getDistanceByMatrix(rawImg, sktImg):

    rowNum = len(rawImg)     #gao and hang
    colNum = len(rawImg[0])  #kuan and lie

    # newImg = np.ones((rowNum, colNum), np.uint8)
    newImg = np.ones((rowNum, colNum))

    for i in range(rowNum):
        for j in range(colNum):
            if rawImg[i][j] > 0:
                dis = rowNum ** 2 + colNum ** 2
                for si in range(rowNum):
                    for sj in range(colNum):
                        # print(sktImg[0][si][sj],'**')
                        if sktImg[0][si][sj] > 0:
                            tempDis = (si - i) ** 2 + (sj - j) ** 2
                            if tempDis < dis:
                                dis = tempDis
                print(dis, 'dis')
                newImg[i][j] = (dis ** 0.5) * 10
    return newImg


def getSkeletonWidthByMatrix(rawImg, outlineImg, sktImg):

    rowNum = len(rawImg)     #gao and hang
    colNum = len(rawImg[0])  #kuan and lie

    newImg = np.ones((rowNum, colNum))
    # newImg = np.ones((rowNum, colNum), np.uint8)

    for i in range(rowNum):
        for j in range(colNum):
            if sktImg[i][j] > 0:
                dis = rowNum**2 + colNum**2
                for si in range(rowNum):
                    for sj in range(colNum):
                        if outlineImg[si][sj] > 0:
                            tempDis = (si - i)**2 + (sj - j)**2
                            if tempDis < dis:
                                dis = tempDis
                newImg[i][j] = dis**0.5


    return newImg


def getWidthByMatrix_x2(rawImg, sktDisImg, sktImg):

    rowNum = len(rawImg)     #gao and hang
    colNum = len(rawImg[0])  #kuan and lie

    newImg = np.ones((rowNum, colNum))
    # newImg = np.ones((rowNum, colNum), np.uint8)

    for i in range(rowNum):
        for j in range(colNum):
            if rawImg[i][j] > 0:
                dis = rowNum**2 + colNum**2
                temp_i = 0
                temp_j = 0
                for si in range(rowNum):
                    for sj in range(colNum):
                        if sktImg[si][sj] > 0:
                            tempDis = (si - i)**2 + (sj - j)**2
                            if tempDis < dis:
                                dis = tempDis
                                temp_i = si
                                temp_j = sj
                newImg[i][j] = sktDisImg[temp_i][temp_j] * 2

    return newImg


def expand(img):
    rowNum = len(img)     #gao and hang
    colNum = len(img[0])  #kuan and lie
    newImg = np.zeros((rowNum, colNum))
    for i in range(rowNum):
        for j in range(colNum):
            if img[i][j] > 0:
                top = i-1
                bot = i+1
                lef = j-1
                rig = j+1
                if i == 0:
                    top = 0
                if i == rowNum-1:
                    bot = rowNum-1
                if j == 0:
                    lef = 0
                if j == colNum-1:
                    rig = colNum-1
                # print(i,j,top,bot,lef,rig)
                newImg[top][lef] = 255
                newImg[top][j] = 255
                newImg[top][rig] = 255
                newImg[i][lef] = 255
                newImg[i][j] = 255
                newImg[i][rig] = 255
                newImg[bot][lef] = 255
                newImg[bot][j] = 255
                newImg[bot][rig] = 255
    return newImg


def preciseProcessDeleteAndCompare(exp, raw, base):
    expandNum = 0
    rowNum = len(base)     #gao and hang
    colNum = len(base[0])  #kuan and lie
    newImg = raw
    # tempImg = np.zeros((rowNum, colNum))
    for i in range(rowNum):
        for j in range(colNum):
            if exp[i][j] > 0:
                if raw[i][j] == 0:
                    if base[i][j] > 0.30:
                        # print(exp[i][j], base[i][j], raw[i][j], i, j)
                        # newImg[i][j] = 255
                        newImg[i][j] = base[i][j]
                        expandNum = expandNum + 1

    # for i in range(rowNum):
    #     for j in range(colNum):
    #         if tempImg[i][j] > 0:
    #             if base[i][j] > 0:
    #                 newImg[i][j] = 255
    #                 # print(exp[i][j],base[i][j],tempImg[i][j],base[i][j],i,j)
    #                 expandNum = expandNum + 1

    return newImg, expandNum


def preciseProcess(mask, img):

    rowNum = len(img)     #gao and hang
    colNum = len(img[0])  #kuan and lie

    # newImg = np.ones((rowNum, colNum), np.uint8)
    newImg = np.zeros((rowNum, colNum))

    outline = getOutlineByMatrix(mask)

    for i in range(rowNum):
        for j in range(colNum):
            if outline[i][j] > 0 and img[i][j] > 0:
                newImg[i][j] = 255
    # return newImg
    expandNum = 1
    while expandNum > 0:
        exp = expand(newImg)
        expandAdd, expandNum = preciseProcessDeleteAndCompare(exp, newImg, img)
        newImg = expandAdd

        # print(expandNum)

    return newImg


def getWidthByMatrix(rawImg, targetImg):

    rowNum = len(rawImg)     #gao and hang
    colNum = len(rawImg[0])  #kuan and lie

    windowSize = 10

    # newImg = np.ones((rowNum, colNum), np.uint8)
    newImg = np.zeros((rowNum, colNum))
    newImg2 = np.zeros((rowNum, colNum))

    ol = getOutlineByMatrix(rawImg)

    for ti in range(rowNum):
        for tj in range(colNum):
            if targetImg[ti][tj] > 0:

                # dis = rowNum^2 + colNum^2
                dis = windowSize**2 + windowSize**2
                for ri in range(2*windowSize+1):
                    for rj in range(2*windowSize+1):
                # for ri in range(rowNum):
                #     for rj in range(colNum):
                        wi = ti-windowSize+ri
                        wj = tj-windowSize+rj
                        if wi < 0:
                            wi = 0
                        if wi >= rowNum:
                            wi = rowNum-1
                        if wj < 0:
                            wj = 0
                        if wj >= colNum:
                            wj = colNum - 1
                        if ol[wi][wj] > 0:
                            tempDis = (wi - ti)**2 + (wj - tj)**2
                        # if rawImg[ri][rj] > 0:
                        #     tempDis = (ri - ti)^2 + (rj - tj)^2
                            if tempDis < dis:
                                dis = tempDis
                newImg[ti][tj] = dis**0.5

                if newImg[ti][tj] < 0.5:
                    newImg[ti][tj] = 0.5

    for ti in range(rowNum):
        for tj in range(colNum):
            if rawImg[ti][tj] > 0:
                # dis = rowNum^2 + colNum^2
                dis = windowSize ** 2 + windowSize ** 2
                mi = 0
                mj = 0
                for ri in range(2 * windowSize + 1):
                    for rj in range(2 * windowSize + 1):
                        wi = ti - windowSize + ri
                        wj = tj - windowSize + rj
                        if wi < 0:
                            wi = 0
                        if wi >= rowNum:
                            wi = rowNum - 1
                        if wj < 0:
                            wj = 0
                        if wj >= colNum:
                            wj = colNum - 1
                        # if ol[wi][wj] > 0:
                        #     tempDis = (ri - ti)^2 + (rj - tj)^2
                        if targetImg[wi][wj] > 0:
                            tempDis = (wi - ti) ** 2 + (wj - tj) ** 2
                            if tempDis < dis:
                                dis = tempDis
                                mi = wi
                                mj = wj
                # newImg2[ti][tj] = dis^0.5 + newImg[ti][tj]
                newImg2[ti][tj] = 2 * newImg[mi][mj]

    newImg3 = np.zeros((rowNum, colNum))
    shangThres = 3
    for ti in range(rowNum):
        for tj in range(colNum):
            if newImg2[ti][tj] >= shangThres:
                newImg3[ti][tj] = 0
                # newImg3[ti][tj] = targetImg[ti][tj]
            else:
                newImg3[ti][tj] = rawImg[ti][tj]

    newImg4 = np.zeros((rowNum, colNum))
    xiaThres = 3
    for ti in range(rowNum):
        for tj in range(colNum):
            if newImg2[ti][tj] < xiaThres:
                newImg4[ti][tj] = 0
                # newImg4[ti][tj] = targetImg[ti][tj]
            else:
                newImg4[ti][tj] = rawImg[ti][tj]

    return newImg2, newImg3, newImg4

img = plt.imread('DRIVE/test/1st_manual/20_manual1.gif')

outline = getOutlineByMatrix(img)
skeleton, dn = deleteByOutline(img, outline)

skeleton2 = skeleton
cnt = 0
while dn > 0:
    ol = getOutlineByMatrix(skeleton2)
    skeleton2, dn = deleteByOutline(skeleton2, ol)
    cnt = cnt + 1
    print(cnt, dn)

ddd, dd2, dd3 = getWidthByMatrix(img, skeleton2)

io.imsave('20_thinmanual1.gif', dd2/255)
io.imsave('20_thickmanual1.gif', dd3/255)

