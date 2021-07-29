import numpy as np
import readCCP
np.set_printoptions(precision=16)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=400)
from matplotlib import  pyplot as plt
#Cmpers = 299702547   #speed of light in CLE
Cmpers = 299792458   #speed of light in void
(Mat,Kalman_Mat_1,Kalman_Mat_2) = readCCP.ReadCCP()    #Receive the parameter from the other function


def LSTS(Mat,Kalman_Mat):
    aij = np.ones(256)      #Set the initial value
    ai = np.ones(256)
    aj = np.ones(256)
    bj = np.zeros(256)
    rou = np.zeros(256)
    taoj = np.zeros(256)
    taoi = np.zeros(256)
    taoi[0] = Mat[0, 0]
    taoj[0] = Kalman_Mat[0, 6]
    l = 1
    rou_a = 0.5
    rou_b = 0.99
    u = 0.5
    reject_num = 0
    #u = float(input('Please input the u (0,1): '))

    while l <= 255:
        taoi[l] = Mat[l,0]
        taoj[l] = Kalman_Mat[l,6]
        k = 1
        temp = 1
        while k <= l:
            temp += k**2
            k += 1
        if l <= 75:       #Set the threshehold to reject
            rou[l] = l**2/temp
            aij[l] = (1 - rou[l])*aij[l-1] + rou[l]*((taoj[l] - taoj[0])/(taoi[l] - taoi[0]))
            at = 1/(1+l)**u
            aj[l] = (1-at*rou_a)*aj[l] + at*rou_a*((aij[l])**-1)*ai[l]
            bj[l] = bj[l] + rou_b*(taoi[l] - taoj[l])
        elif 75 < l <= 100:
            aij[l] = aij[l-1]
            aj[l] = aj[l-1]
            bj[l] = bj[l-1]
            reject_num += 1
        elif 100 < l <= 200:
            rou[l] = l**2/temp
            aij[l] = (1 - rou[l])*aij[l-1] + rou[l]*((taoj[l] - taoj[100])/(taoi[l] - taoi[100]))
            at = 1/(1+l)**u
            aj[l] = (1-at*rou_a)*aj[l] + at*rou_a*((aij[l])**-1)*ai[l]
            bj[l] = bj[l] + rou_b*(taoi[l] - taoj[l])
        elif 200 < l <= 225:
            aij[l] = aij[l - 1]
            aj[l] = aj[l - 1]
            bj[l] = bj[l - 1]
            reject_num += 1
        elif 225 < l :
            rou[l] = l**2/temp
            aij[l] = (1 - rou[l])*aij[l-1] + rou[l]*((taoj[l] - taoj[225])/(taoi[l] - taoi[225]))
            at = 1/(1+l)**u
            aj[l] = (1-at*rou_a)*aj[l] + at*rou_a*((aij[l])**-1)*ai[l]
            bj[l] = bj[l] + rou_b*(taoi[l] - taoj[l])
        print('the seq number is :' + str(l) + '    aij: ' + str(aij[l]) + '     aj:' + str(aj[l]))
        print('the offset bj is :  ' + str(bj[l]))
        l += 1
    print('the reject number is:  ' + str(reject_num))

    return aij,aj,bj


def Compare_drifting(Mat,Kalman_Mat,aj):
    List_Tx = np.array([])                       #Prepare the empty list and the initial value
    List_Rx = np.array([])
    List_Rx_cali = np.array([])
    List_Rx_aij = np.array([])
    i = 0
    aij_sum_error = 0
    cali_sum_error = 0


    while i < 255:
        delta_Tx = Mat[i + 1, 0] - Mat[i, 0]
        delta_Rx = Kalman_Mat[i+1,3] - Kalman_Mat[i,3]
        delta_Rx_cali = delta_Rx / Kalman_Mat[i, 7]

        if i <=100:
            delta_Rx_aij = delta_Rx/aij[75]
        elif 100 < i <= 225:
            delta_Rx_aij = delta_Rx / aij[200]
        elif 225 < i:
            delta_Rx_aij = delta_Rx / aij[255]

        List_Tx = np.append(List_Tx, delta_Tx)              #Put the data into list
        List_Rx = np.append(List_Rx, delta_Rx)
        List_Rx_cali = np.append(List_Rx_cali, delta_Rx_cali)
        List_Rx_aij = np.append(List_Rx_aij,delta_Rx_aij)
        cali_sum_error += (delta_Tx - delta_Rx_cali)**2           #Calculate the square error
        aij_sum_error += (delta_Tx - delta_Rx_aij)**2
        i += 1

    print('the square of kalman drifting error is :   ' + str(cali_sum_error/256))
    print('the square of the LSTS drifting error is :   ' + str(aij_sum_error/256))
    x = np.arange(0,List_Tx.size)
    plt.plot(x, List_Tx, color='black')
    plt.plot(x, List_Rx, color='red')
    plt.plot(x, List_Rx_cali, color='green')
    plt.plot(x,List_Rx_aij,color='blue')
    plt.show()




if __name__  ==  '__main__':
    (aij,aj,bj) = LSTS(Mat,Kalman_Mat_2)
    Compare_drifting(Mat,Kalman_Mat_2,aj)
