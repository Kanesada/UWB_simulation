import random
import numpy as np
from scipy.optimize import leastsq
from scipy.optimize import least_squares
from math import sqrt
np.set_printoptions(precision=20)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=400)


S = np.array([[0, 0, 3.2], [-4, 7, 0], [8, 0, 0], [15, 15, 2.5]])
#S = np.array([[0, 0, 10], [10, 0, 0], [0, 10, 10], [10, 10, 0]])
numAnchors = S.shape[0]
tagposition = np.array([5, 5, 5])

doa = np.zeros(numAnchors)
doa_true = np.zeros(numAnchors)
error_list = np.array([])


def distance(a,b):
    d = sqrt(np.inner(a - b, a - b))
    return d

def distance_sqr(a,b):
    d = np.inner(a - b, a - b)
    return d


def Generate_NLOS_doa():
    for i in range(numAnchors):
        doa_true[i] = distance(tagposition, S[i])
        bias = random.randint(1, 5) / 10
        doa[i] = doa_true[i] + random.gauss(0, bias) + bias
    # print('The true doa is: ' + str(doa_true))
    # print('The doa measurement is: ' + str(doa))
    return

def Generate_LOS_doa():
    for i in range(numAnchors):
        doa_true[i] = distance(tagposition, S[i])
        doa[i] = doa_true[i] + random.gauss(0, 0.15)
    # print('The true doa is: ' + str(doa_true))
    # print('The doa measurement is: ' + str(doa))
    return

def LLSE():
    A = np.zeros(shape=(numAnchors - 1, S.shape[1]))
    b = np.zeros(numAnchors - 1)
    for i in range(numAnchors - 1):
        A[i] = 2*(S[i+1] - S[0])
        b[i] = doa[i+1]*doa[i+1] - doa[0]*doa[0] + distance_sqr(S[0], 0) - distance_sqr(S[i+1], 0)
    result = np.linalg.pinv(A) @ b
    #print('the result from LLSE is: ' + str(-1*result))
    return -1*result


def f(x):
    fx = np.zeros(numAnchors)
    for i in range(numAnchors):
        #fx[i] = distance(np.append(x, set_height), S[i]) - doa[i]
        fx[i] = distance(x, S[i]) - doa[i]
    return fx**2


def Get_result(Guess):
    #result = leastsq(f, Guess)[0]
    # if result[0] * Guess[0] > 0:   #Constraint
    #     return result
    # else:
    #     return -1*result

    result = least_squares(f, Guess, loss='soft_l1', f_scale=0.1)
    return result['x']



def Draw3D():
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    ax = plt.axes(projection='3d')
    ax.scatter3D(tagposition[0], tagposition[1], tagposition[2], color='blue')
    for i in range(numAnchors):
        ax.scatter3D(S[i, 0], S[i, 1], S[i, 2],  color='red')
    #ax.scatter3D(result[0], result[1], result[2], color='green')
    ax.scatter3D(result2[0], result2[1], result2[2], color='purple')
    plt.show()



def One_hundred():
    global error_list, result2
    for i in range(100):
        Generate_NLOS_doa()
        result = LLSE()
        result2 = Get_result(result)
        #print('The result from LM is : ' + str(result2))
        error = distance(result2, tagposition)
        error_list = np.append(error_list, error)

    RMSE = error_list.mean(0)
    print('\n' + '均方根误差: ' + str(RMSE) + "米")
    Draw3D()

def One_time():
    Generate_NLOS_doa()
    result = LLSE()
    global result2
    result2 = Get_result(result)
    print('The result from LM is : ' + str(result2))
    Draw3D()



#One_time()    # Test one time
One_hundred()   #Test one hundred times
