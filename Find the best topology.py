import numpy as np
import matplotlib.pyplot as plt

#####1. 定义室内空间布局
A  = np.array([0,1])
B = np.array([2,1])
C = np.array([0,0])
D = np.array([2,0])

####2.定义基站坐标矩阵初值

P0 = np.zeros((2,4))

###3.生成基站的采样点栅格
a = np.linspace(0,2,11)
b = np.linspace(0,1,6)
A,B = np.meshgrid(a,b)


###4.生成标签采样点栅格
c = np.linspace(0,2,101)
d = np.linspace(0,1,51)
C,D = np.meshgrid(c,d)

plt.style.use('ggplot')
plt.plot(A,B,marker = '.', linestyle = 'none')
plt.show()

plt.style.use('ggplot')
plt.plot(C,D,marker = '.', linestyle = 'none')
plt.show()




