from scipy.optimize import leastsq
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np
np.set_printoptions(precision=20)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=400)
p = r'Ori.csv'

refAnchorCoords = np.array([2.496,2.454])
S = np.mat([[2.842, 7.979], [9.875, 6.089], [7.200, 1.049]])

with open(p) as f:
    data = np.loadtxt(f,str,delimiter = ",")
    data = data[:,3:]
    data = data.astype('float')
    DDOA = data[:,:3]
    RESULT = data[:,3:]
    #DDOA[:, [0, 2]] = DDOA[:, [2, 0]]    #sort the ddoa as S2 S3 S4



