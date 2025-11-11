import numpy as np
from skimage import measure

a = np.zeros((3,3))
label, region_nums = measure.label(a, background=0, return_num=True, connectivity=2)
region = measure.regionprops(label)

for i in range(0):
    print(1)