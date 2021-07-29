import random
import numpy as np
from scipy.optimize import leastsq
from math import sqrt
np.set_printoptions(precision=20)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=400)


S = np.array([[3, 0, 0], [0, 4, 0], [0, 0, 5], [0, 0, 0]])
numAnchors = S.shape[0]
doa = np.array([6.403, 5.831, 5, 7.0711])
tagposition = np.array([3, 4, 5])
doa_true = np.zeros(numAnchors)


def distance(a, b):
    d = sqrt(np.inner(a - b, a - b))
    return d

def distance_sqr(a, b):
    d = np.inner(a - b, a - b)
    return d


for i in range(numAnchors):
    doa_true[i] = distance(tagposition, S[i])
print(doa_true)



def LLSE():
    A = np.zeros(shape=(numAnchors - 1, S.shape[1]))
    b = np.zeros(numAnchors - 1)
    for i in range(numAnchors - 1):
        A[i] = 2*(S[i+1] - S[0])
        b[i] = doa[i+1]*doa[i+1] - doa[0]*doa[0] + distance_sqr(S[0], 0) - distance_sqr(S[i+1], 0)
    result = np.linalg.pinv(A) @ b
    print(-1*result)
    return -1*result

def f(x):
    fx = np.zeros(numAnchors)
    for i in range(numAnchors - 1):
        fx[i] = distance(x, S[i]) - doa[i]
    return fx**2


def Draw3D():
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    ax = plt.axes(projection='3d')
    ax.scatter3D(tagposition[0], tagposition[1], tagposition[2], color='blue')
    for i in range(numAnchors):
        ax.scatter3D(S[i, 0], S[i, 1], S[i, 2],  color='red')
    ax.scatter3D(result1[0], result1[1], result1[2], color='yellow')
    ax.scatter3D(result2[0], result2[1], result2[2], color='green')
    plt.show()




#Generate_NLOS_doa()
result1 = LLSE()
result2 = leastsq(f, result1)[0]
print(result2)
Draw3D()


