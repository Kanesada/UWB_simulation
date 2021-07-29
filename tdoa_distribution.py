import numpy as np
from matplotlib import  pyplot as plt
from scipy import stats
from scipy.stats import kstest
import seaborn as sns
import ReadTDOA

(error1, error2, error3) = ReadTDOA.Read_TDOA()
a = error1
sns.distplot(a)
plt.show()
print(a.size)
plt.plot(range(a.shape[0]), a)
plt.show()