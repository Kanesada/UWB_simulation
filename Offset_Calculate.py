import numpy as np
from scipy.optimize import leastsq
from math import sqrt

c = 299792458   #speed of light in void

####   For anc1   #########
tx1 = 4.526701459209735e-01
rx1 = 4.591181381460337e-01
x11 = 1.000000000417185e+00
x01 = 4.591181375323372e-01


tx2 = 5.575821692395333e+00
rx2 = 5.582268470286834e+00
x12 = +9.999999873616173e-01
x02 = +5.582268470409322e+00

tx3 = 2.812896788549179e+00
rx3 = 2.819341378220778e+00
x13 = +9.999999833725242e-01
x03 = +2.819341378263141e+00


tx4 = 6.136243423164563e+00
rx4 = 6.142684566446940e+00
x14 = +9.999999811554575e-01
x04 = +6.142684566451492e+00


offset1 = tx1 + 1/c - rx1/x11
offset2 = tx2 + 2/c - rx2/x12
offset3 = tx3 + sqrt(2)/c - rx3/x13

l1 = (c*(rx1/x11 - tx1 + offset1))**2
l2 = (c*(rx2/x12 - tx2 + offset2))**2
l3 =(c*(rx3/x13 - tx3 + offset3))**2
print(offset1)
print(offset2)
print(offset3)


def func(i):
    x,y = i
    return np.asarray((
            x**2 + y**2 - l1,
            (x-1)**2 + y**2 - l2,
            x**2 + (y-1)**2 - l3,
        ))

root = leastsq(func, np.asarray((0,0,)))
print(root[0])