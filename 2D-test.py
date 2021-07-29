from scipy.optimize import fsolve
from scipy.optimize import leastsq
from math import sqrt
import numpy as np
np.set_printoptions(precision=20)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=400)

refAnchorCoords = np.array([0,0])
S = np.mat([[2,0],[2,2],[0,2]])
tagposition = np.array([1.2,1.2,1])
Guess = np.array([1,1])
ddoa = np.array([0.0691,-0.5691,0.0640])    #measurement
#ddoa = np.array([0.0579,-0.5657,0.0579])         #true value
#ddoa = np.array([0.1591,-0.5791,0.1740])          #NLOS




def func(x):
    x0 = float(x[0])
    x1 = float(x[1])


def f(x):
    x0 = float(x[0])
    x1 = float(x[1])
    d0 = sqrt((x0 - refAnchorCoords[0])**2 + (x1 - refAnchorCoords[1])**2)
    return [
        sqrt((x0 - S[0,0])**2 + (x1 - S[0,1])**2 ) - d0 - ddoa[0],
        sqrt((x0 - S[1,0])**2 + (x1 - S[1,1])**2 ) - d0 - ddoa[1],
        sqrt((x0 - S[2,0])**2 + (x1 - S[2,1])**2 ) - d0 - ddoa[2]
    ]



result1 = leastsq(f, Guess)



print(result1)
