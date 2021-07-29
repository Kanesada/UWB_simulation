import seaborn as sns
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
np.set_printoptions(precision=20)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=400)
p = r'firstMeasurement.csv'  #Put the Log into a csv file
t = r'SecondMeasurement.csv'
###### Set the True Value ##########
refAnchorCoords = np.array([3.598, 3.222, 2.10])     #Master Anchor's Coordinate
S = np.mat([[0.270, 3.245, 2.1], [0.300, -0.979, 2.1], [3.637, -1.002, 2.1]])  #Slave Anchor's Coordinates
numAnchors = S.shape[0] + 1


###### Get the Error #########
with open(p) as f1:
    data1 = np.loadtxt(f1, str,delimiter=",")
    data1 = data1.astype('float')
    data1 = data1[10:, :]
    # print(data)
    # print(data.shape[0])

with open(t) as f2:
    data2 = np.loadtxt(f2, str, delimiter=",")
    data2 = data2.astype('float')
    data2 = data2[40:, :]
    print(data2.shape[0])


####### First measurement: X Y Z #########
    x = np.arange(0, data1.shape[0])
    plt.subplot(221)
    plt.title('The X axis coordinates')
    plt.plot(x, data1[:, 0])
    plt.subplot(222)
    plt.title('The Y axis coordinates')
    plt.plot(x, data1[:, 1])
    plt.subplot(223)
    plt.title('The Z axis coordinates')
    plt.plot(x, data1[:, 2])
    plt.show()

###### Show Distribution ######
    plt.subplot(221)
    sns.distplot(data1[:, 0])
    plt.title('The X distribution of first measurement')
    plt.subplot(222)
    sns.distplot(data1[:, 1])
    plt.title('The Y distribution of first measurement')
    plt.subplot(223)
    sns.distplot(data1[:, 2])
    plt.title('The Z distribution of first measurement')
    plt.show()

####### Second measurement: X Y Z #########
    x = np.arange(0, data2.shape[0])
    plt.subplot(221)
    plt.title('The X axis coordinates')
    plt.plot(x, data2[:, 0])
    plt.subplot(222)
    plt.title('The Y axis coordinates')
    plt.plot(x, data2[:, 1])
    plt.subplot(223)
    plt.title('The Z axis coordinates')
    plt.plot(x, data2[:, 2])
    plt.show()

    ###### Show Distribution ######
    plt.subplot(221)
    sns.distplot(data2[:, 0])
    plt.title('The X distribution of second measurement')
    plt.subplot(222)
    sns.distplot(data2[:, 1])
    plt.title('The Y distribution of second measurement')
    plt.subplot(223)
    sns.distplot(data2[:, 2])
    plt.title('The Z distribution of second measurement')
    plt.show()


###### Multilateration Result #########
    ax = plt.axes(projection='3d')
    plt.title('The Localization Result')
    ax.scatter3D(refAnchorCoords[0], refAnchorCoords[1], refAnchorCoords[2], color='red')  #The Anchor
    for i in range(numAnchors - 1):
        ax.scatter3D(S[i, 0], S[i, 1], S[i, 2], color='red')  #The Anchor
    ax.scatter3D(data1[:, 0], data1[:, 1], data1[:, 2], color='purple')   # The measurment
    ax.scatter3D(data2[:, 0], data2[:, 1], data2[:, 2], color='green')  # The measurment
    plt.show()


