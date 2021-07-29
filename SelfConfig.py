from scipy.optimize import leastsq
import numpy as np
from math import sqrt
from scipy.optimize import dual_annealing
np.set_printoptions(precision=20)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=400)

M1 = np.array([2.496,2.454])        #  Master anchor's coordinates
MS2 = 5.53                          #  The distance between Master and slaves
MS3 = 8.23
MS4 = 4.90
S23 = 7.29
S24 = 8.19
S34 = 5.73
distance_list = np.array([5.53, 8.23, 4.90, 7.29, 8.19, 5.73])
numAnchors = 4


def Check():
    #Check the number of anchors and distance measurement
    n_distance = distance_list.shape[0]
    numAnchors_ = int((1 + sqrt(1 + 8 * n_distance)) / 2)
    if (numAnchors != numAnchors_):
        print("Error:The number of anchors and distance not matched\n")
        return 0
    else:
        return 1


def distance(a, b):
    return sqrt(np.inner(a - b, a - b))

def distance_sqr(a, b):
    return np.inner(a - b, a - b)

def distance_sqr_4(a, b, c, d):
    return (a-c)**2 + (b - d)**2

def f(x):
    fx = np.zeros(distance_list.shape[0])

    fx[0] = distance_sqr_4(x[0], x[1], 0, 0) - distance_list[0]**2
    fx[1] = distance_sqr_4(x[2], x[3], 0, 0) - distance_list[1]**2
    fx[2] = distance_sqr_4(x[4], x[5], 0, 0) - distance_list[2]**2
    fx[3] = distance_sqr_4(x[0], x[1], x[2], x[3]) - distance_list[3]**2
    fx[4] = distance_sqr_4(x[0], x[1], x[4], x[5]) - distance_list[4]**2
    fx[5] = distance_sqr_4(x[2], x[3], x[4], x[5]) - distance_list[5]**2
    return fx@fx.T
'''
def f(x):
    fx = np.zeros(distance_list.shape[0])
    ori = np.array([0, 0])
    slave = np.zeros(int(distance_list.shape[0]/2))
    slave[0] = np.array(x[0], x[1])
    slave[1] = np.array(x[2], x[3])
    slave[2] = np.array(x[4], x[5])

    fx[0] = distance_sqr(ori, slave[0]) - distance_list[0]**2
    fx[1] = distance_sqr(ori, slave[1]) - distance_list[1]**2
    fx[2] = distance_sqr(ori, slave[2]) - distance_list[2]**2
    fx[3] = distance_sqr(slave[0], slave[1]) - distance_list[3]**2
    fx[4] = distance_sqr(slave[0], slave[2]) - distance_list[4]**2
    fx[5] = distance_sqr(slave[1], slave[2]) - distance_list[5]**2
    return fx@fx.T'''


def SA_func(x):
    x0 = float(x[0])
    x1 = float(x[1])
    x2 = float(x[2])
    x3 = float(x[3])
    x4 = float(x[4])
    x5 = float(x[5])

    fx = np.array([x0**2 + x1**2 - MS2**2,
        x2**2 + x3**2 - MS3**2,
        x4**2 + x5**2 - MS4**2,
        (x0 - x2)**2 + (x1 - x3)**2 - S23**2,
        (x2 - x4)**2 + (x3 - x5)**2 - S34**2,
        (x0 - x4)**2 + (x1 - x5)**2 - S24**2])

    return fx@fx.T




def Draw3D():
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    ax = plt.axes(projection='3d')
    ax.scatter3D(M1[0], M1[1], color='red')
    for i in range(result_SA.shape[0]):
        ax.scatter3D(result_SA[i, 0], result_SA[i, 1],  color='purple')
    plt.show()

def Check(result):
    anchor = np.vstack((M1, result))
    d = np.zeros(6)
    d[0] = distance(anchor[0], anchor[1])
    d[1] = distance(anchor[0], anchor[2])
    d[2] = distance(anchor[0], anchor[3])
    d[3] = distance(anchor[1], anchor[2])
    d[4] = distance(anchor[1], anchor[3])
    d[5] = distance(anchor[2], anchor[3])
    print(d)





bounds = [(-20, 20), (-20, 20), (-20, 20), (-20, 20), (-20, 20), (-20, 20)]
result_DA = dual_annealing(SA_func, bounds)
result_SA_ = result_DA.x
result_SA = result_SA_.reshape(3, 2)+M1

bounds2 = [(-20, 20), (-20, 20), (-20, 20), (-20, 20), (-20, 20), (-20, 20)]
result_DA2 = dual_annealing(f, bounds2)
result_SA2_ = result_DA2.x
result_SA2 = result_SA2_.reshape(3, 2)+M1



print('\n')
print(result_SA)
Draw3D()
Check(result_SA)

print(result_SA2)
Check(result_SA2)

