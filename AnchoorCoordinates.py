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


Guess = np.array([2.842, 7.979,  9.875, 6.089,  7.200, 1.049])           #The guess value
#Guess = np.zeros(6)

def distance(a, b):
    d = sqrt(np.inner(a - b, a - b))
    return d

def f(x):
    x0 = float(x[0])
    x1 = float(x[1])
    x2 = float(x[2])
    x3 = float(x[3])
    x4 = float(x[4])
    x5 = float(x[5])

    fx = [
        x0**2 + x1**2 - MS2**2,
        x2**2 + x3**2 - MS3**2,
        x4**2 + x5**2 - MS4**2,
        (x0 - x2)**2 + (x1 - x3)**2 - S23**2,
        (x2 - x4)**2 + (x3 - x5)**2 - S34**2,
        (x0 - x4)**2 + (x1 - x5)**2 - S24**2
    ]
    return fx

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

    #print(fx)
    return fx@fx.T




def Draw3D():
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    ax = plt.axes(projection='3d')
    ax.scatter3D(M1[0], M1[1], color='red')
    for i in range(result.shape[0]):
        ax.scatter3D(result[i, 0], result[i, 1],  color='green')
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

#result = fsolve(f, [1, 1, 1.09])
result1 = leastsq(f, Guess)
array = np.array([*result1])
result = array[0].reshape(3, 2) + M1

bounds = [(-20, 20), (-20, 20), (-20, 20), (-20, 20), (-20, 20), (-20, 20)]
result_DA = dual_annealing(SA_func, bounds)
result_SA_ = result_DA.x
result_SA = result_SA_.reshape(3, 2)+M1

#print(result1)
print(result)
print('\n')
print(result_SA)
Draw3D()
Check(result_SA)

#print(type(array))

