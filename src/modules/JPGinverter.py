import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

# from IMGpath import IMGdir_labtop
# from IMGpath import IMGdir_lab


# 정해진 디렉토리의 이미지를 List로 return
def getIMGS():
    IMGdir = "C:\\Users\\Hilal\\pycharmProjects\\softacturatorModel\\testIMG"
    os.chdir(IMGdir) # 해당 폴더로 이동
    IMGfiles = os.listdir(IMGdir) # 해당 폴더에 있는 파일 이름을 리스트 형태로 받음

    dataList = []
    for img in IMGfiles:
        dataList.append(cv2.imread(img, 0))
        # testing 이미지 출력
        plt.imshow(dataList[0])
        plt.show()
    return(np.array(dataList))

