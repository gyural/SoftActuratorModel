#main Function
from modules.JPGinverter import getIMGS
import numpy as np
dataSets = getIMGS()
print(dataSets[0].shape)

