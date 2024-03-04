import cv2
import numpy as np
from matplotlib import pyplot as plt

sampleIMG = cv2.imread(f'C:\Users\Hilal\pycharmProjects\softacturatorModel\testIMG\testIMG.jpg'
                       , cv2.IMREAD_GRAYSCALE)
plt.imshow(sampleIMG, cmap="gray"), plt.axis("off")
plt.show()
print("JPG Inverting!!!")