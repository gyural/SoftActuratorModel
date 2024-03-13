import numpy as np
import random
img_size = 800 * 600
channel = 1

# 3차원 numpy array가 들어왔을때 seperating
# @return {list} Yvalue전처리 한 값
def img_sep_main(datas):
    Yvalues = []
    for i in range(len(datas)):
        Yvalues.append(image_seperator(datas[i]))
    return Yvalues

#img는 800 x 600의 이미지 데이터
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
