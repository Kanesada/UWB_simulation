import random
import numpy as np
from scipy.optimize import leastsq
from math import sqrt
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

np.set_printoptions(precision=20)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=400)

####### 在此处设置基站坐标 （x ，y ，z） ##########
#S = np.array([[0, 0, 3.2], [-4, 7, 2.8], [8, 0, 3.1], [5, 5, 3.0]])  # 基站不良好布局
S = np.array([[0, 0, 3.0], [-4, 7, 3.0], [8, 0, 3], [5, 5, 3.0]])  # 基站布置在一个平面
#S = np.array([[0, 0, 0], [10, 0, 10], [0, 10, 10], [10, 10, 0]])    # 基站良好布局
#S = np.array([[-0.2, 0.2, 0.1], [0.2, 0.2, 0.2], [-0.2, -0.2, 0.3], [0.2, -0.2, 0.4]])   # 变态布局 暂时无法处理
#S = np.array([[0, 0, 0], [6, 0, 0], [0, 5, 0], [3.5, 3, 0], [3, 2.5, 0.5]])   # 对比东南大学论文布局
#S = np.array([[0.2, 10.6, 2.84], [3.676, 11.648, 0], [4.156, 10.6, 3.08], [8.231, 10.6, 3.16]])  #会议室基站布局
numAnchors = S.shape[0]

######## 在此处设置标签坐标（x ，y ，z） ##########
#tagposition = np.array([2.5, 1, 0.98])
tagposition = np.array([7.457, 3.658, 0.78])

doa = np.zeros(numAnchors)
doa_true = np.zeros(numAnchors)
error_list = np.array([])
LLSE_height_list = np.array([])
LM_height_list = np.array([])

# 此函数计算两点间距离
def distance(a, b):
    d = sqrt(np.inner(a - b, a - b))
    return d

# 此函数计算两点间距离的平方
def distance_sqr(a,b):
    d = np.inner(a - b, a - b)
    return d


# 此函数用来生成非视距下的测量噪声
# bias代表非视距误差偏置
# random.gauss(0, 0.4)中的0.4代表测距的随机误差为0.4m
def Generate_NLOS_doa():
    for i in range(numAnchors):
        doa_true[i] = distance(tagposition, S[i])
        bias = random.randint(0, 5) / 10  # 非视距造成0~50cm的随机误差偏置
        doa[i] = doa_true[i] + random.gauss(0, 0.4) + bias  # 除了随机误差偏置 非视距也会使测量噪声增大
    # print('The true doa is: ' + str(doa_true))
    # print('The doa measurement is: ' + str(doa))
    return

# 此函数用来生成可视距下的测量噪声
# random.gauss(0, 0.15)中的0.15代表测距的随机误差为0.15m
def Generate_LOS_doa():
    for i in range(numAnchors):
        doa_true[i] = distance(tagposition, S[i])
        doa[i] = doa_true[i] + random.gauss(0, 0.15)
    # print('The true doa is: ' + str(doa_true))
    # print('The doa measurement is: ' + str(doa))
    return

# LLSE算法计算全局初值的粗定位结果
def LLSE():
    A = np.zeros(shape=(numAnchors - 1, S.shape[1]))
    b = np.zeros(numAnchors - 1)
    for i in range(numAnchors - 1):
        A[i] = 2*(S[i+1] - S[0])
        b[i] = doa[i+1]*doa[i+1] - doa[0]*doa[0] + distance_sqr(S[0], 0) - distance_sqr(S[i+1], 0)
    #result = np.linalg.pinv(A) @ b
    ATA = A.T@A
    regular_matrix = np.eye(ATA.shape[0], ATA.shape[0])
    #result = np.linalg.inv(ATA) @ A.T @ b
    result = np.linalg.inv(ATA + regular_matrix) @A.T @b
    #print('the result from LLSE is: ' + str(-1*result))
    return -1*result

# 定义LSMR法中的目标函数矩阵
def f(x):
    fx = np.zeros(numAnchors)
    for i in range(numAnchors):
        #fx[i] = distance(np.append(x, set_height), S[i]) - doa[i]
        fx[i] = distance(x, S[i]) - doa[i]
    return fx**2

# 调用非线性优化库中的LSMR法求解
# 当基站布局不良好时，各基站高度比较接近，基站矩阵的第三列的秩近似不足，若使用常见算法计算误差大。
# LSMR属于信赖域反射算法的一种，通过引入正则化缓解了jacobian阵中秩不足导致病态解的问题。
def Get_result(Guess):
    result_robust = least_squares(f, Guess, method='trf', tr_options={'regularize': 0}, tr_solver='lsmr')['x']
    #result_robust = least_squares(f, Guess, method='lm')['x']
    return result_robust


# 绘制三维可视化示意图
def Draw3D():
    ax = plt.axes(projection='3d')
    ax.scatter3D(tagposition[0], tagposition[1], tagposition[2], color='blue')
    for i in range(numAnchors):
        ax.scatter3D(S[i, 0], S[i, 1], S[i, 2],  color='red')
    #ax.scatter3D(result[0], result[1], result[2], color='green')
    ax.scatter3D(result2[0], result2[1], result2[2], color='purple')
    plt.xlabel('Red:Anchor  Blue:True tag')
    plt.show()

# 统计100次测量
def One_hundred():
    global error_list, result2, LM_height_list, LLSE_height_list
    for i in range(100):    # 进行100次测量
        Generate_LOS_doa()
        result = LLSE()
        z_max = np.max(S[:, 2])
        Guess = np.array([result[0], result[1], 0.5*z_max])
        result2 = Get_result(Guess)
        #print('The result from LM is : ' + str(result2))
        error = distance(result2, tagposition)
        error_list = np.append(error_list, error)
        LLSE_height_list = np.append(LLSE_height_list, result[2])
        LM_height_list = np.append(LM_height_list, result2[2])

    RMSE = error_list.mean(0)
    LLSE_RMSE = abs(LLSE_height_list - tagposition[2])
    LLSE_RMSE = LLSE_RMSE.mean(0)
    LM_RMSE = abs(LM_height_list - tagposition[2])
    LM_RMSE = LM_RMSE.mean(0)
    LLSE_Mean_Height = LLSE_height_list.mean(0)
    LM_Mean_Height = LM_height_list.mean(0)


    print('\n' + 'LSMR法整体均方根误差: ' + str(RMSE) + "米")

    print('\n' + 'LLSE法粗估计高度均方根误差: ' + str(LLSE_RMSE) + "米")
    print('\n' + 'LLSE法粗估计求得平均高度: ' + str(LLSE_Mean_Height) + "米")
    print('\n')
    print('\n' + 'LM法最终高度均方根误差: ' + str(LM_RMSE) + "米")
    print('\n' + 'LM法最终求得平均高度: ' + str(LM_Mean_Height) + "米")

    x = np.arange(0, LLSE_height_list.shape[0])
    plt.xlabel('Measurement number : ' + str(LLSE_height_list.shape[0]))
    plt.title('The Z axis error of LLSE without regularization')
    plt.plot(x, LLSE_height_list - tagposition[2], 'ob', markersize=2)
    plt.show()

    x = np.arange(0, LM_height_list.shape[0])
    plt.xlabel('Measurement number : ' + str(LM_height_list.shape[0]))
    plt.title('The Z axis error of regularized nonlinear fitting')
    plt.plot(x, LM_height_list - tagposition[2], 'ob', markersize=2)
    plt.show()

    result2[2] = LM_Mean_Height
    Draw3D()

# 进行单次测量
def One_time():
    Generate_LOS_doa()
    result = LLSE()
    global result2
    result2 = Get_result(result)
    print('The result from LSMR is : ' + str(result2))
    Draw3D()



####### 主函数 ##########
if  __name__  ==  '__main__':
    #One_time()     # 进行单次测量
    One_hundred()  # 进行100次测量 其中高度值取定位结果平均数

