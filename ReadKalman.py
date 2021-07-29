import numpy as np
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=400)
from matplotlib import  pyplot as plt
#Cmpers = 299702547   #speed of light in CLE
Cmpers = 299792458   #speed of light in void

with open('P4_255.log','r')as file:

    All_Lines = file.readlines()

    def ReadKalman():

        Kalman_Mat_1 = np.zeros(shape=(256,8))
        Kalman_Mat_2 = np.zeros(shape=(256, 8))

        for line in All_Lines:
            if ":CS_Kalman" in line :

                Kalmanline = line
                list1 = Kalmanline.split(' ')
                anc = int(list1[2])
                seq = int(list1[-7])
                TX_ = float(list1[-6])
                RX = float(list1[-5])
                delta_TX = float(list1[-4])
                Measurement_Error = float(list1[-3])
                X0 = float(list1[-2])
                X1 = list1[-1]
                X1 = X1.split('\\')
                X1 = float(X1[0])




                if anc == 1 :
                    Kalman_Mat_1[seq,0] = anc
                    Kalman_Mat_1[seq, 1] = seq
                    Kalman_Mat_1[seq, 2] = TX_
                    Kalman_Mat_1[seq, 3] = RX
                    Kalman_Mat_1[seq, 4] = delta_TX
                    Kalman_Mat_1[seq, 5] = Measurement_Error
                    Kalman_Mat_1[seq, 6] = X0
                    Kalman_Mat_1[seq, 7] = X1

                if anc == 2 :
                    Kalman_Mat_2[seq,0] = anc
                    Kalman_Mat_2[seq, 1] = seq
                    Kalman_Mat_2[seq, 2] = TX_
                    Kalman_Mat_2[seq, 3] = RX
                    Kalman_Mat_2[seq, 4] = delta_TX
                    Kalman_Mat_2[seq, 5] = Measurement_Error
                    Kalman_Mat_2[seq, 6] = X0
                    Kalman_Mat_2[seq, 7] = X1
                print(list1)


        print(Kalman_Mat_1)
        np.savetxt('kalman1.csv', Kalman_Mat_1, delimiter=',')
        np.savetxt('kalman2.csv', Kalman_Mat_2, delimiter=',')
        print('     ')
        print(Kalman_Mat_2)







    ReadKalman()

