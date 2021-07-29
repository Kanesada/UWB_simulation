import numpy as np
import csv
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

np.set_printoptions(precision=20)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=400)

refAnchorCoords = np.array([0, 0, 5])
S = np.array([[10, 0, 0], [10, 10, 5], [0, 10, 0]])
numAnchors = S.shape[0] + 1

data = r'LM_track.csv'
result = r'gramSG_result.csv'
py_result = r'SG_track.csv'

LM = np.loadtxt(data, str, delimiter=",")
SG_C = np.loadtxt(result, str, delimiter=",")
SG_Py = np.loadtxt(py_result, str, delimiter=",")

LM = LM.astype('float')
SG_C = SG_C.astype('float')
SG_Py = SG_Py.astype('float')

ax = plt.axes(projection='3d')
ax.scatter3D(refAnchorCoords[0], refAnchorCoords[1], refAnchorCoords[2], color='red')
for i in range(numAnchors - 1):
    ax.scatter3D(S[i, 0], S[i, 1], S[i, 2], color='red')
#ax.scatter3D(LM[:, 0], LM[:, 1], color='purple')
ax.scatter3D(SG_C[:, 0], SG_C[:, 1], color='green')
ax.scatter3D(SG_Py[:, 0], SG_Py[:, 1], color='blue')
plt.show()
