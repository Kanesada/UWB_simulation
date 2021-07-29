import random
import numpy as np
from scipy.optimize import leastsq
from math import sqrt
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.optimize import dual_annealing
"""

A0（0,10）                          A1(10,10)
         X                          X
         
    -------------------------------------   Obstacle（Wooden Door）                  
             Tag3   *  
                    |    
         Tag0  *----|----*  Tag1
                    |
                    *  Tag2

         X                          X
A2(0,0)                          A3(10,0)

"""
Statics = True
np.set_printoptions(precision=20)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=400)

####### 在此处设置基站坐标 （x ，y ，z） ##########
A = np.array([[0, 10], [10, 10], [0, 0], [10,0]])  # Four anchors
S_x = A[:, 0] # 各基站X值
S_y = A[:, 1] # 各基站Y值
# S_z = A[:, 2] # 各基站Z值
numAnchors = A.shape[0] + 1


L = 0.2
tag = np.zeros((4, 2))
tag[0] = np.array([2, 8])
tag[1] = np.array([tag[0][0]+L, tag[0][1]])
tag[2] = np.array([tag[0][0]+0.5*L, tag[0][1]-0.5*L])
tag[3] = tag[2] + np.array([0, L])
numTag = tag.shape[0]
print('Tag position: ' + str(tag))

ddoa = np.zeros((tag.shape[0], numAnchors - 1))
ddoa_true = np.zeros((tag.shape[0], numAnchors - 1))


NLOS_A = np.array([0, 1])   # Anchor 0 and Anchor 1 are in NLOS condition
LOS_A = np.array([2, 3])    # Anchor 2 and Anchor 3 are in NLOS condition

def distance(a, b):
    d = sqrt(np.inner(a - b, a - b))
    return d

def distance_sqr(a, b):
    d = np.inner(a - b, a - b)
    return d

##### Distance Measurement #####
def Generate_NLOS_ddoa():
    """
          A1  A2   A3  A4
    Tag1  d1  d2   d3  d4
    Tag2  d1  d2   d3  d4
    """
    bias = 0.15
    for i in range(tag.shape[0]):
        for j in NLOS_A:
            ddoa_true[i, j] = distance(tag[i], A[j])
            ddoa[i, j] = ddoa_true[i, j] + random.gauss(0, 0.1) + bias
            #ddoa[i, j] = ddoa_true[i, j] + bias

        for k in LOS_A:
            ddoa_true[i, k] = distance(tag[i], A[k])
            ddoa[i, k] = ddoa_true[i, k] + random.gauss(0, 0.1)
            #ddoa[i, k] = ddoa_true[i, k]
    return ddoa, ddoa_true

"""
       |---------Tag coord------------|---Anchor bias---|
     x=[x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10，x11]

"""
def Calculation_Function(x):
    # fx = np.zeros(2*numAnchors+numTag-1)
    # for i in range(numTag):
    #     for j in range(numAnchors-1):
    #         fx[4*i+j] = (ddoa[i, j] - x[2*numTag+j])**2 - distance_sqr(np.array(x[2*i],x[2*i+1]), A[j])
    # fx[2*numAnchors+numTag-2] = L**2 - distance_sqr(np.array(x[0],x[1]), np.array(x[2],x[3]))
    # #print(fx)
    # return fx@fx.T

    fx = np.zeros(numTag*numAnchors+numTag+2)
    fx[0] = (ddoa[0, 0] - x[8])**2 - distance_sqr(np.array([x[0],x[1]]), A[0])
    fx[1] = (ddoa[0, 1] - x[9]) ** 2 - distance_sqr(np.array([x[0], x[1]]), A[1])
    fx[2] = (ddoa[0, 2] - x[10]) ** 2 - distance_sqr(np.array([x[0], x[1]]), A[2])
    fx[3] = (ddoa[0, 3] - x[11]) ** 2 - distance_sqr(np.array([x[0], x[1]]), A[3])

    fx[4] = (ddoa[1, 0] - x[8])**2 - distance_sqr(np.array([x[2],x[3]]), A[0])
    fx[5] = (ddoa[1, 1] - x[9]) ** 2 - distance_sqr(np.array([x[2], x[3]]), A[1])
    fx[6] = (ddoa[1, 2] - x[10]) ** 2 - distance_sqr(np.array([x[2], x[3]]), A[2])
    fx[7] = (ddoa[1, 3] - x[11]) ** 2 - distance_sqr(np.array([x[2], x[3]]), A[3])

    fx[8] = (ddoa[2, 0] - x[8])**2 - distance_sqr(np.array([x[4],x[5]]), A[0])
    fx[9] = (ddoa[2, 1] - x[9]) ** 2 - distance_sqr(np.array([x[4], x[5]]), A[1])
    fx[10] = (ddoa[2, 2] - x[10]) ** 2 - distance_sqr(np.array([x[4], x[5]]), A[2])
    fx[11] = (ddoa[2, 3] - x[11]) ** 2 - distance_sqr(np.array([x[4], x[5]]), A[3])

    fx[12] = (ddoa[3, 0] - x[8])**2 - distance_sqr(np.array([x[6],x[7]]), A[0])
    fx[13] = (ddoa[3, 1] - x[9]) ** 2 - distance_sqr(np.array([x[6], x[7]]), A[1])
    fx[14] = (ddoa[3, 2] - x[10]) ** 2 - distance_sqr(np.array([x[6], x[7]]), A[2])
    fx[15] = (ddoa[3, 3] - x[11]) ** 2 - distance_sqr(np.array([x[6], x[7]]), A[3])

    fx[16] = (L**2 - distance_sqr(np.array(x[0],x[1]), np.array(x[2], x[3])))
    fx[17] = (0.5*L**2 - distance_sqr(np.array(x[0], x[1]), np.array(x[4], x[5])))
    fx[18] = (0.5 * L ** 2 - distance_sqr(np.array(x[2], x[3]), np.array(x[4], x[5])))
    fx[19] = (L ** 2 - distance_sqr(np.array(x[4], x[5]), np.array(x[6], x[7])))
    fx[20] = (0.5 * L ** 2 - distance_sqr(np.array(x[0], x[1]), np.array(x[6], x[7])))
    fx[21] = (0.5 * L ** 2 - distance_sqr(np.array(x[2], x[3]), np.array(x[6], x[7])))
    return fx@fx.T


def Test_Function(x):
    # fx = np.zeros(2*numAnchors+numTag-1)
    # for i in range(numTag):
    #     for j in range(numAnchors-1):
    #         fx[4*i+j] = (ddoa[i, j] - x[2*numTag+j])**2 - distance_sqr(np.array(x[2*i],x[2*i+1]), A[j])
    # fx[2*numAnchors+numTag-2] = L**2 - distance_sqr(np.array(x[0],x[1]), np.array(x[2],x[3]))
    # #print(fx)
    # return fx@fx.T

    fx = np.zeros(numTag*numAnchors+numTag+2)
    fx[0] = (ddoa[0, 0] - x[8])**2 - distance_sqr(np.array([x[0],x[1]]), A[0])
    fx[1] = (ddoa[0, 1] - x[9]) ** 2 - distance_sqr(np.array([x[0], x[1]]), A[1])
    fx[2] = (ddoa[0, 2] - x[10]) ** 2 - distance_sqr(np.array([x[0], x[1]]), A[2])
    fx[3] = (ddoa[0, 3] - x[11]) ** 2 - distance_sqr(np.array([x[0], x[1]]), A[3])

    fx[4] = (ddoa[1, 0] - x[8])**2 - distance_sqr(np.array([x[2],x[3]]), A[0])
    fx[5] = (ddoa[1, 1] - x[9]) ** 2 - distance_sqr(np.array([x[2], x[3]]), A[1])
    fx[6] = (ddoa[1, 2] - x[10]) ** 2 - distance_sqr(np.array([x[2], x[3]]), A[2])
    fx[7] = (ddoa[1, 3] - x[11]) ** 2 - distance_sqr(np.array([x[2], x[3]]), A[3])

    fx[8] = (ddoa[2, 0] - x[8])**2 - distance_sqr(np.array([x[4],x[5]]), A[0])
    fx[9] = (ddoa[2, 1] - x[9]) ** 2 - distance_sqr(np.array([x[4], x[5]]), A[1])
    fx[10] = (ddoa[2, 2] - x[10]) ** 2 - distance_sqr(np.array([x[4], x[5]]), A[2])
    fx[11] = (ddoa[2, 3] - x[11]) ** 2 - distance_sqr(np.array([x[4], x[5]]), A[3])

    fx[12] = (ddoa[3, 0] - x[8])**2 - distance_sqr(np.array([x[6],x[7]]), A[0])
    fx[13] = (ddoa[3, 1] - x[9]) ** 2 - distance_sqr(np.array([x[6], x[7]]), A[1])
    fx[14] = (ddoa[3, 2] - x[10]) ** 2 - distance_sqr(np.array([x[6], x[7]]), A[2])
    fx[15] = (ddoa[3, 3] - x[11]) ** 2 - distance_sqr(np.array([x[6], x[7]]), A[3])

    fx[16] = L**2 - distance_sqr(np.array(x[0],x[1]), np.array(x[2], x[3]))
    fx[17] = (0.5*L**2 - distance_sqr(np.array(x[0], x[1]), np.array(x[4], x[5])))
    fx[18] = (0.5 * L ** 2 - distance_sqr(np.array(x[2], x[3]), np.array(x[4], x[5])))
    fx[19] = L ** 2 - distance_sqr(np.array(x[4], x[5]), np.array(x[6], x[7]))
    fx[20] = (0.5 * L ** 2 - distance_sqr(np.array(x[0], x[1]), np.array(x[6], x[7])))
    fx[21] = (0.5 * L ** 2 - distance_sqr(np.array(x[2], x[3]), np.array(x[6], x[7])))
    return fx



if Statics:
    numStatics = 1000
    ddoa_statics = np.zeros((numStatics, ddoa.shape[0], ddoa.shape[1]))
    for i in range(numStatics):
        Generate_NLOS_ddoa()
        ddoa_statics[i, :, :] = ddoa
    #print(ddoa_statics)
    ddoa_mean = np.mean(ddoa_statics, axis=0)
    ddoa_std = np.std(ddoa_statics, axis=0)
    ddoa = ddoa_mean
    #print("ddoa variance is : " + str(ddoa_std))
else:
    Generate_NLOS_ddoa()

bounds = [(-20, 20), (-20, 20), (-20, 20), (-20, 20),(-20, 20), (-10, 20), (-20, 20), (-10, 20),\
          (-0.01, 0.5), (-0.01, 0.5), (-0.01, 0.5), (-0.01, 0.5)]
ret_DA = dual_annealing(Calculation_Function, bounds=bounds, seed=1234, maxiter=1000)
print(ret_DA)

# Guess = np.array([2, 8, 2, 8, 2, 8, 2, 8, 0,0,0,0])
# result = least_squares(Test_Function, Guess)
# print(result)
