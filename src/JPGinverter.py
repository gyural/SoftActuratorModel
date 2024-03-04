import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

IMGdir = "C:\\Users\\Hilal\\pycharmProjects\\softacturatorModel\\testIMG"
os.chdir(IMGdir) # 해당 폴더로 이동
IMGfiles = os.listdir(IMGdir) # 해당 폴더에 있는 파일 이름을 리스트 형태로 받음

dataList = []
for img in IMGfiles:
    dataList.append(IMGdir + "\\" + cv2.imread(img))

plt.show(dataList[0])