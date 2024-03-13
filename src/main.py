#main Function
import torch
import numpy as np
from regression.simple.IMGseperating import img_sep_main
from modules.JPGinverter import getIMGS
from matplotlib import pyplot as plt


#### IMG Testing ####
# dataList = getIMGS()
# # print(np.shape(dataList[1]))
# plt.imshow(dataList[1])
# plt.show()
#### ML Testing ####
# 225 x 225 x sampleNum
X_image = torch.randn(100, 600, 800)

Y_values = img_sep_main(np.array(X_image))
print(Y_values)


