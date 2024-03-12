import numpy as np
import random
img_size = 225 * 225
channel = 1

Yvalues = []
def img_sep_main(datas):
    for i in range(len(datas)):
        Yvalues.append(image_seperator(datas[i]))
    return Yvalues

#img는 225 x 225의 배열임
def image_seperator(img):
    value = np.sum(img)
    if value > img_size * 0.7:
        return random.randint(40,50)
    elif value > img_size * 0.3:
        return random.randint(30, 40)
    elif value > img_size * 0:
        return random.randint(20, 30)
    elif value > img_size * -0.3:
        return random.randint(10, 20)
    else:
        return random.randint(0, 10)
