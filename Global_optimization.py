from scipy.optimize import leastsq
from scipy.optimize import shgo
from scipy.optimize import dual_annealing
import numpy as np
from math import sqrt
np.set_printoptions(precision=20)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=400)

Master_coord = np.array([2.496, 2.454])        #  Master anchor's coordinate
numAnchors = 4
#distance_list = np.array([5.53, 8.23, 4.90, 7.29, 8.19, 5.73])
distance_list = np.array([])
S = np.array([[2.496, 2.454], [2.0555959424995858, 7.966321501176125], [9.293194877732166, 7.094284917154981],
               [7.338699215689116, 1.7080358374341698]])


def Send_TWR_Instruct(T, R):
    global distance_list
    d = distance(S[T], S[R])
    distance_list = np.append(distance_list, d)
    print('   Anchor%s <---> Anchor%s  %f' % (T, R, d))

def Do_TWR():
    D = np.zeros(shape=(numAnchors, numAnchors))
    n = 0
    print('Start the Two Way Ranging: ')
    for i in range(numAnchors):
        for j in range(i+1, numAnchors):
            Send_TWR_Instruct(i, j)
            D[i, j] = distance_list[n]
            n += 1
    print('\nThe distance matrix is : ')
    print(str(D) + '\n')
    #print('The distance list is: ' + str(distance_list))
    return D


#####################################################################

#Slave_coord = Master_coord + np.array([distance_list[0], 0])  #The first slave anchor's coordinate
#Guess = np.array([[9.875, 6.089],  [7.200, 1.049]])         #The guess value
Guess = np.zeros(2*(numAnchors-2))        #The guess value


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
    d = sqrt(np.inner(a - b, a - b))
    return d


def Cal_Anchors_function(x):
    fx = np.zeros(distance_list.shape[0] - 1)
    fx[0] = (Master_coord[0]-x[0])**2 + (Master_coord[1]-x[1])**2 - distance_list[1]**2
    fx[1] = (Slave_coord[0] - x[2]) ** 2 + (Slave_coord[1] - x[3]) ** 2 - distance_list[2] ** 2
    fx[2] = (Slave_coord[0] - x[0])**2 + (Slave_coord[1] - x[1])**2 - distance_list[3]**2
    fx[3] = (Slave_coord[0] - x[2])**2 + (Slave_coord[1] - x[3])**2 - distance_list[4]**2
    fx[4] = (x[0] - x[2])**2 + (x[1] - x[3])**2 - distance_list[5]**2
    return fx@fx.T



'''
def SA_function(x):
    fx = np.zeros(distance_list.shape[0] - 1)
    for n in range(int(0.5*(numAnchors * (numAnchors -1)))):
        if n == 1:
            Master_coord[0] - 


    return fx**2
'''

def Cal_Anchors_function2(x):
    fx = np.zeros(distance_list.shape[0] - 1)
    n = 0
    for i in range(numAnchors-1):
        if i == 0:
            for j in range(i + 2, numAnchors):
                fx[n] = distance(Master_coord, x[j-2]) - distance_list[n+1]
                n += 1
        elif i == 1:
            for j in range(i + 1, numAnchors):
                fx[n] = distance(Slave_coord, x[j-2]) - distance_list[n+1]
                n += 1
        else:
            for j in range(i + 1, numAnchors):
                fx[n] = distance(x[i-2], x[i-1]) - distance_list[n+1]
                n += 1
    return sum(abs(fx))

def Get_Coords():
    result = leastsq(Cal_Anchors_function, Guess)[0]
    result = result.reshape(numAnchors-2, Master_coord.shape[0])
    final = np.vstack((Master_coord, Slave_coord, result))
    print(final)
    return final

def Draw2D(result):
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    ax = plt.axes(projection='3d')
    for i in range(result.shape[0]):
        ax.scatter3D(result[i, 0], result[i, 1],  color='red')
    plt.show()


Do_TWR()
if Check():
    #Coords = Get_Coords()
    Slave_coord = Master_coord + np.array([distance_list[0], 0])  # The first slave anchor's coordinate
    bounds = [(-20, 20), (-20, 20), (-20, 20), (-20, 20)]
    result_DA = dual_annealing(Cal_Anchors_function, bounds)
    result = result_DA.x
    result = result.reshape(numAnchors - 2, Master_coord.shape[0])
    Coords = np.vstack((Master_coord, Slave_coord, result))
    print(Coords)
    Draw2D(Coords)



'''
Master<-->Anchoe2  8.23
Master<-->Anchoe3  4.9
Slave<-->Anchoe2  7.29
Slave<-->Anchoe3  8.19
Anchor2<-->Anchor3 5.73

'''
