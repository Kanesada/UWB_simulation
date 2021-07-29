import random
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

np.set_printoptions(precision=20)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=400)

a = np.array([.1, .7, .9, .7, .8, .5, -1*.3])
a_ = np.array([.1, .7, .9, .7, .8, .5, -1*.3, 0.1])
b = savgol_filter(a, 7, 2)
b_ = savgol_filter(a_, 7, 2)
print(b)
print(b_)
