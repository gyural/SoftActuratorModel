#main Function
import torch
import numpy as np
from regression.simple.IMGseperating import img_sep_main
X_image = torch.randn(100, 1, 255, 255)
img_arr = np.array(X_image)

# 225 x 225 x sampleNum
image_list=[]

for i in range(img_arr.shape[0]):
    image_list.append(img_arr[i][0])
Y_values = img_sep_main(np.array(image_list))
print(Y_values)


