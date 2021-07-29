import numpy as np
from matplotlib import  pyplot as plt
from scipy import stats
from scipy.stats import kstest
import seaborn as sns
#Cmpers = 299702547   #speed of light in CLE
Cmpers = 299792458   #speed of light in void
list_error1 = np.array([])  # prepare an empty array for error
list_error2 = np.array([])  # prepare an empty array for error
list_error3 = np.array([])  # prepare an empty array for error
list_kalman_error = np.array([])
with open('test11.log','r')as file:
    All_Lines = file.readlines()
    def Read_Kalman_Error():    # a function to read clock syn kalman error
        global list_kalman_error
        kalman_num = 0
        Splited_lines = All_Lines[62:]    #select data from no.63 row to filter out the rejected kalman data
        for line in All_Lines:
            if ":CS_Kalman" in line :
                kalman_line = line[9:19]      #the row include kalman data
                kalman_error = float(line[97:119] )   #the error of clock syn kalman filter
                time = line[:9]
                kalman_num += 1
                list_kalman_error = np.append(list_kalman_error,kalman_error)
                print(time + "   :" + str(kalman_error) + "    " + str(kalman_num))
        list_kalman_error_distance = Cmpers * list_kalman_error
        x = np.arange(0,list_kalman_error.size)
        plt.subplot(221)
        plt.title('the clock syn error ')
        plt.xlabel('the sequeence number')
        plt.ylabel('the kalman error(s)')
        plt.plot(x,list_kalman_error,'ob',markersize=0.5)
        plt.subplot(223)
        plt.xlabel('the sequeence number')
        plt.ylabel('the sync error contributed to distance error(m)')
        plt.plot(x,list_kalman_error_distance,'ob',markersize=0.5)
        plt.savefig('Clock syn error.jpg')
        plt.show()

        list_kalman_error_square = list_kalman_error_distance ** 2
        kalman_len = len(list_kalman_error_distance)
        sum = 0
        for i in list_kalman_error_square:
            sum += i
        t = sum/kalman_len
        #d = t*(Cmpers**2)
        print(t)
        #print(d**0.5)
        test = kstest(list_kalman_error,'norm')
        print(test)

        return list_kalman_error_distance



    def Read_TDOA():
        global list_error1
        global list_error2
        global list_error3
        multilaterate_num = 0
        blink_num = 0
        list_ddoa = np.empty(1000, dtype=float)  # prepare an empty array for ddoa results
        for line in All_Lines:
            '''if 'BLINK' in line:
                multilaterate_num += 1
                blink = line
                list_b = blink.split(':')
                anc = list_b[2].split(' ')[2]
                FP = list_b[-1].split('\\')
                print(anc)
                print(FP)'''
            if 'Multilaterate' in line:
                multilaterate_num += 1
                multilaterate = line
                list1 = multilaterate.split('     ')    # to split the string
                list2 = list1[1]
                list2 = list2.split(' ')
                tdoa_result = list2[1:5]
                tdoa = [0,0,0,0]    # a list prepared to load tdoa value
                for a in tdoa_result:    #sort the tdoa reults
                    if a[0] == '0':
                        tdoa[0] = 0
                    elif a[0] == '1':
                        c = a.split(':')
                        tdoa_value = float(c[1])
                        tdoa[1] = tdoa_value
                    elif a[0] == '2':
                        c = a.split(':')
                        tdoa_value = float(c[1])
                        tdoa[2] = tdoa_value
                    elif a[0] == '3':
                        c = a.split(':')
                        tdoa_value = float(c[1])
                        tdoa[3] = tdoa_value
                d = np.array(tdoa)
                ddoa = d*Cmpers
                distance_erro1 = abs(ddoa[1]) - 0.02   #the difference between measurement and true value
                distance_erro2 = abs(ddoa[2]) - 0.22
                distance_erro3 = abs(ddoa[1]) - 0.20
                if abs(distance_erro1) < 1:               #reject the giant value   set the threshehold as 1m
                    list_error1 = np.append(list_error1, distance_erro1)
                if abs(distance_erro2) < 1:  # reject the giant value   set the threshehold as 1m
                    list_error2 = np.append(list_error2, distance_erro2)
                if abs(distance_erro3) < 1:  # reject the giant value   set the threshehold as 1m
                    list_error3 = np.append(list_error3, distance_erro3)
                current_time = list2[-1]
                current_time = current_time.split('\n')
                current_time = current_time[0]
                #print(line)
                #print(list2)
                '''print('the unsorted TDOA results are: ' + str(tdoa_result))
                print('the sorted tdoa results are: ' + str(tdoa))   #print the tdoa results in float type list
                print('the distance difference of arrival(sorted) is : ' + str(ddoa))
                print('the  error of anc1 is : ' + str(distance_erro1))
                print('the  error of anc2 is : ' + str(distance_erro2))
                print('the  error of anc3 is : ' + str(distance_erro3))
                print('current time: ' + str(current_time))
                print('the multilateration number is ' + str(multilaterate_num))
                print('\n')'''
        #print(list_error1)
        reject_num1 = abs(multilaterate_num - list_error1.size)
        reject_num2 = abs(multilaterate_num - list_error2.size)
        reject_num3 = abs(multilaterate_num - list_error3.size)
        print('the number of all error is : ' + str(list_error1.size))
        print('rejected number of anc1: ' + str(reject_num1))
        print('rejected number of anc2: ' + str(reject_num2))
        print('rejected number of anc3: ' + str(reject_num3))
        mean_error1 = list_error1.mean()
        mean_error2 = list_error2.mean()
        mean_error3 = list_error3.mean()
        print('the mean error of anc1 is : ' + str(mean_error1))
        print('the mean error of anc2 is : ' + str(mean_error2))
        print('the mean error of anc3 is : ' + str(mean_error3))
        x1 = np.arange(0,list_error1.size)
        x2 = np.arange(0, list_error2.size)
        x3 = np.arange(0, list_error3.size)
        '''plt.title('the error of anc1')
        plt.xlabel('the sequeence number')
        plt.ylabel('the distance error')
        plt.plot(x,list_error1)'''

        #sns.distplot(list_error1)
        if __name__ == '__main__':
            plt.subplot(221)
            plt.xlabel('the mean error of anc1 is : ' + str(mean_error1))
            plt.plot(x1, list_error1)
            plt.subplot(222)
            plt.xlabel('the mean error of anc2 is : ' + str(mean_error2))
            plt.plot(x2, list_error2)
            plt.subplot(223)
            plt.xlabel('the mean error of anc3 is : ' + str(mean_error3))
            plt.plot(x3, list_error3)

            print('the var is ' + str(np.var(list_error1)))



            plt.savefig(current_time.replace(':','') + '.jpg')     #replace the ':' in filename
            plt.show()
        return list_error1, list_error2, list_error3
    #Read_Kalman_Error()
    Read_TDOA()

