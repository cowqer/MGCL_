import numpy as np
from PIL import Image

label = np.array(Image.open('/data/seekyou/Data/DLRSD/Labels/baseballdiamond/baseballdiamond11.png'))
print(np.unique(label))
