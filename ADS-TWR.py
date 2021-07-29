import random
import numpy as np
from math import sqrt
np.set_printoptions(precision=20)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=400)
C = 299702547
delay_true = 0.000000258114
delay = delay_true * 0.98
drifting = 1.000000067176924      # Assum A is true time, Ta = drifting * Tb

def distance(a, b):
    d = sqrt(np.inner(a - b, a - b))
    return d

Anchor = np.array([[0, 0], [0, 10]])
Tf_true = distance(Anchor[0], Anchor[1])/C
bias = random.randint(0, 5) / 10
Tf = (distance(Anchor[0], Anchor[1]) + random.gauss(0, 0.05) + bias) / C
#Tf = (distance(Anchor[0], Anchor[1]) + random.gauss(0, 0.01)) / C
print(Tf)

Ra = 2*Tf + delay_true
Rb = (2*Tf + delay_true) / drifting

def TWR():
    d = 0.5*(Ra - delay) * C
    return d

def ADS_TWR():
    d1 = Ra*Rb - delay*delay
    d2 = Ra + Rb + 2*delay
    return (d1/d2)*C

print(TWR())
print(ADS_TWR())
