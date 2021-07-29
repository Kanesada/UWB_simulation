import numpy as np
from scipy.optimize import leastsq
from math import sqrt
import ReadCCP
np.set_printoptions(precision=20)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=400)
from matplotlib import  pyplot as plt

#Cmpers = 299702547   #speed of light in CLE
Cmpers = 299792458   #speed of light in vacuum
Time_Unit = (1.0/499.2e6/128.0)     #Device time unit
Clock_Period = 1099511627776   #(0x10000000000LL)
Clock_Period_Sec = Time_Unit*Clock_Period
(Mat,Kalman_Mat_1,Kalman_Mat_2) = ReadCCP.ReadCCP()    #Receive the parameter from the other function

def Run_Kalman(Mat,Kalman_Mat):
    List_T = np.array([])  ##  Get T interval of master and slaves
    List_R = np.array([])
    i = 0
    while i < 255:
        delta_T = Mat[i + 1, 0] - Mat[i, 0]
        delta_R = Mat[i + 1, 1] - Mat[i, 1]
        List_T = np.append(List_T, delta_T)
        List_R = np.append(List_R, delta_R)
        i += 1

    offset_Guess = -0.644*10**-3        # Init the matrix
    drifting_Guess = 1
    S = np.transpose(np.mat([[offset_Guess,drifting_Guess]]))
    M = np.mat([[1,1],[1,1]])
    B = np.mat([[0,0],[-1,1]])
    BT = np.transpose(B)
    n1=n2=v1=v2 = 0.1    ########
    v = np.mat([v2,v1])  ###
    u = np.transpose(np.mat([n1,n2]))   ###
    Qv = np.identity(2)
    thetaf = 10**-9   #####
    thetan = 10**-9  ######
    Qv = thetaf*Qv
    Qu = np.identity(2)
    H = np.mat([1,0])   ###
    HT = np.transpose(H)
    C = np.mat([-1,1])   ###
    CT = np.transpose(C)
    Qu = thetan*Qu
    K = np.mat([0.5,0.5])


    k = 0
    while k < 255:
        A = np.mat([[1, List_R[k]], [0, 1]])
        AT = np.transpose(A)
        ###### Prediction ##########
        S = A*S     # State Function
        M = A*M*AT + B*Qv*BT  # Uncertainty exploration
        K = M*HT*(np.linalg.inv(C*Qu*CT + H*M*HT))     # Kalman gain

        ##### Correction ##########
        #x = H*S + C*u    #   Measurement
        x = Mat[k,0] - Mat[k,1]
        S = S + K*(x - H*S)   # State update   #########
        M = (np.identity(2) - K*H)*M
        k += 1
    print(S)




if __name__  ==  '__main__':
    Run_Kalman(Mat,Kalman_Mat_1)
