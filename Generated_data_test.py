import numpy as np
import random
from matplotlib import pyplot as plt
import seaborn as sns
fig,ax=plt.subplots()

np.random.seed(4) #设置随机数种子
bias = random.randint(2, 5) / 10
Gaussian = np.random.normal(bias, 0.01, 1000)
ax.hist(Gaussian, bins=25, histtype="stepfilled", density=True, alpha=0.6)
sns.kdeplot(Gaussian, shade=True)
plt.show()
plt.plot(range(1000), Gaussian)
plt.show()