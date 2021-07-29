import random
import numpy as np
from scipy.optimize import leastsq
from scipy.optimize import least_squares
from math import sqrt
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
np.set_printoptions(precision=20)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=400)

####### 在此处设置基站坐标 （x ，y ，z） ##########
#S = np.array([[0, 0, 3.2], [-4, 7, 2.8], [8, 0, 3.1], [5, 5, 3.0]])  #基站不良好布局
S = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [10, 10, 0],[0, 0, 5], [10, 0, 5],
             [0, 10, 5], [10, 10, 5], [0, 0, 10], [10, 0, 10], [0, 10, 10], [10, 10, 10]])    #基站良好布局
numAnchors = S.shape[0]

######## 在此处设置标签坐标（x ，y ，z） ##########
tagposition = np.array([3, 2, 3.7])

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
def Generate_NLOS_doa(bias_list):
    for i in range(numAnchors):
        doa_true[i] = distance(tagposition, S[i])
        doa[i] = doa_true[i] + random.gauss(0, 0.15+bias_list[i]) + bias_list[i]
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

# LLSE算法计算粗定位结果
def LLSE():
    A = np.zeros(shape=(numAnchors - 1, S.shape[1]))
    b = np.zeros(numAnchors - 1)
    for i in range(numAnchors - 1):
        A[i] = 2*(S[i+1] - S[0])
        b[i] = doa[i+1]*doa[i+1] - doa[0]*doa[0] + distance_sqr(S[0], 0) - distance_sqr(S[i+1], 0)
    #result = np.linalg.pinv(A) @ b
    ATA = A.T@A
    regular_matrix = np.eye(ATA.shape[0], ATA.shape[0])     # 引入正则化矩阵，处理基站布局不良好的情况
    result = np.linalg.inv(ATA) @ A.T @ b                  # 无正则化矩阵，无法处理基站布局不良好的情况
    #result = np.linalg.inv(ATA + regular_matrix) @A.T @b
    #print('the result from LLSE is: ' + str(-1*result))
    return -1*result



# 定义LM法中的目标函数矩阵
def f(x):
    fx = np.zeros(numAnchors)
    for i in range(numAnchors):
        #fx[i] = distance(np.append(x, set_height), S[i]) - doa[i]
        fx[i] = distance(x, S[i]) - doa[i]
    return fx**2

# LM法求解
def Get_result(Guess):
    result = leastsq(f, Guess)[0]
    return result

def Get_robust_result(Guess):
    result_robust = least_squares(f, Guess, loss='cauchy', f_scale=0.02)['x']
    #result_robust = least_squares(f, Guess, method='lm', loss='linear', f_scale=0.05)['x']

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
    error1_list = np.array([])
    error_robust_list = np.array([])
    bias_list = np.array([])


    NLOS_Anchor_number = 4
    for i in range(numAnchors):
        if (numAnchors > NLOS_Anchor_number) and (i >= NLOS_Anchor_number):
            bias = 0
        else:
            bias = random.randint(20, 50) / 100
        bias_list = np.append(bias_list, bias)
    print('各基站测距误差偏置为： ' + str(bias_list))

    for i in range(1000):    # 进行100次测量
        Generate_NLOS_doa(bias_list)    # 生成测量噪声
        result = LLSE()
        result2 = Get_result(result)
        result_robust = Get_robust_result(result)
        #print('The result from LM is : ' + str(result2))
        error1 = distance(result, tagposition)
        error2 = distance(result2, tagposition)
        error_robust =  distance(result_robust, tagposition)
        error1_list = np.append(error1_list, error1)
        error2_list = np.append(error_list, error2)
        error_robust_list = np.append(error_robust_list, error_robust)
        LLSE_height_list = np.append(LLSE_height_list, result[2])
        LM_height_list = np.append(LM_height_list, result2[2])

    RMSE1 = error1_list.mean(0)
    RMSE2 = error2_list.mean(0)
    RMSE_robust = error_robust_list.mean(0)
    LLSE_RMSE = abs(LLSE_height_list - tagposition[2])
    LLSE_RMSE = LLSE_RMSE.mean(0)
    LM_RMSE = abs(LM_height_list - tagposition[2])
    LM_RMSE = LM_RMSE.mean(0)
    LLSE_Mean_Height = LLSE_height_list.mean(0)
    LM_Mean_Height = LM_height_list.mean(0)

    print('\n' + 'LLSE整体均方根误差: ' + str(RMSE1) + "米")
    print('\n' + 'LM整体均方根误差: ' + str(RMSE2) + "米")
    print('\n' + '鲁棒回归整体均方根误差: ' + str(RMSE_robust) + "米")

    print('\n' + 'LLSE法粗估计高度均方根误差: ' + str(LLSE_RMSE) + "米")
    print('\n' + 'LLSE法粗估计求得平均高度: ' + str(LLSE_Mean_Height) + "米")
    print('\n')
    print('\n' + 'LM法最终高度均方根误差: ' + str(LM_RMSE) + "米")
    print('\n' + 'LM法最终求得平均高度: ' + str(LM_Mean_Height) + "米")

    x = np.arange(0, LLSE_height_list.shape[0])
    plt.xlabel('Measurement number : ' + str(LLSE_height_list.shape[0]))
    plt.title('The Z axis error of LLSE')
    plt.plot(x, LLSE_height_list - tagposition[2], 'ob', markersize=2)
    plt.show()

    x = np.arange(0, LM_height_list.shape[0])
    plt.xlabel('Measurement number : ' + str(LM_height_list.shape[0]))
    plt.title('The Z axis error of LM')
    plt.plot(x, LM_height_list - tagposition[2], 'ob', markersize=2)
    plt.show()

    result2[2] = LM_Mean_Height
    Draw3D()

# 进行单次测量
def One_time():
    bias_list = np.array([])
    for i in range(numAnchors):
        bias = random.randint(0, 5) / 10
        bias_list = np.append(bias_list, bias)
    print('各基站测距误差偏置为： ' + str(bias_list))
    Generate_NLOS_doa(bias_list)
    #Generate_LOS_doa()   # 生成测量噪声
    result = LLSE()
    print('The result of LLSE is : ' + str(result))
    global result2
    result2 = Get_result(result)
    print('The result from LM is : ' + str(result2))
    Draw3D()



####### 主函数 ##########
if  __name__  ==  '__main__':
    #One_time()     # 进行单次测量
    One_hundred()  # 进行100次测量 其中高度值取定位结果平均数

