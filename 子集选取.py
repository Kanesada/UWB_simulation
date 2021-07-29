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

# S = np.array([[0.2, 10.6, 2.84], [8.0, 11.648, 3], [0.5, 11.648, 3.08], [8.231, 10.6, 3.16]])  #会议室基站布局

S = np.array([[0, 0, 0],[10, 0, 5],[0, 10, 5],[10, 10, 0],  [0,10,0],[10,10,5],[10,0,0],[0,0,5],
              [0,10,20],[10,10,10],[10,0,20],[0,0,10],  [0,10,25],[10,10,15],[10,0,25],[0,0,15],
              [0,10,10],[10,10,30],[10,0,10],[0,0,30],  [0,10,15],[10,10,35],[10,0,15],[0,0,35],
              [0,10,40],[10,10,20],[10,0,40],[0,0,20],  [0,10,45],[10,10,25],[10,0,45],[0,0,25],
              [0,10,30],[10,10,40],[10,0,30],[0,0,40],  [0,10,35],[10,10,45],[10,0,35],[0,0,45]])    # 基站良好布局
numAnchors = S.shape[0]

######## 在此处设置标签坐标（x ，y ，z） ##########
tagposition = np.array([2.5, 2, 0.98])

doa = np.zeros(numAnchors)
doa_true = np.zeros(numAnchors)
error_list = np.array([])
error_LLSE_list = np.array([])
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
def LLSE(lower=0, upper=numAnchors-1):
    A = np.zeros(shape=(upper-lower, S.shape[1]))
    b = np.zeros(upper-lower)
    j = lower
    for i in range(upper - lower):
        A[i] = 2*(S[j+1] - S[lower])
        b[i] = doa[j+1]*doa[j+1] - doa[lower]*doa[lower] + distance_sqr(S[lower], 0) - distance_sqr(S[j+1], 0)
        j += 1
    #result = np.linalg.pinv(A) @ b
    ATA = A.T@A
    regular_matrix = np.eye(ATA.shape[0], ATA.shape[0])
    result = np.linalg.inv(ATA) @ A.T @ b
    #result = np.linalg.inv(ATA + regular_matrix) @A.T @b
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
    #result_robust = least_squares(f, Guess, method='trf', tr_options={'regularize': 1}, tr_solver='lsmr')['x']
    result_robust = least_squares(f, Guess, method='trf',loss='cauchy', f_scale=0.05,
                                  tr_options={'regularize': 1}, tr_solver='lsmr')['x']
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
    test_number = 1000
    global error_list, result2, LM_height_list, LLSE_height_list, error_LLSE_list
    error_LLSE_sub_list_1 = np.array([])
    error_LLSE_sub_list_2 = np.array([])
    LLSE_sum_list = np.array([])

    for i in range(test_number):    # 进行100次测量
        Generate_LOS_doa()
        result = LLSE()
        result_sub_1_0 = LLSE(0, 20)    # 子集0
        result_sub_1_2 = LLSE(1, 21)  # 子集1
        result_sub_1_3 = LLSE(5, 25)  # 子集2
        result_sub_1_4 = LLSE(10, 30)  # 子集3
        result_sub_2 = 0.4918*result_sub_1_0 + 0.2185*result_sub_1_2 + 0.1972*result_sub_1_3 + 0.0923*result_sub_1_4
        Guess = np.array([result[0], result[1], 0])
        result2 = Get_result(Guess)
        #print('The result from LM is : ' + str(result2))
        if i <= test_number:
            #LLSE_sum_list = np.append(LLSE_sum_list, result_sub_1)
            LLSE_sum_list = np.append(LLSE_sum_list, result)


        error = distance(result2, tagposition)             # 计算两算法整体定位误差
        error_LLSE = distance(result, tagposition)
        error_LLSE_sub_1 = distance(result_sub_1_0, tagposition)
        error_LLSE_sub_2 = distance(result_sub_2, tagposition)

        error_list = np.append(error_list, error)         # 生成两算法整体定位误差列表
        error_LLSE_list = np.append(error_LLSE_list, error_LLSE)
        error_LLSE_sub_list_1 = np.append(error_LLSE_sub_list_1, error_LLSE_sub_1)
        error_LLSE_sub_list_2 = np.append(error_LLSE_sub_list_2, error_LLSE_sub_2)

        LLSE_height_list = np.append(LLSE_height_list, result[2])      # 生成两算法高度结果列表
        LM_height_list = np.append(LM_height_list, result2[2])


    LLSE_sum_list = LLSE_sum_list.reshape(-1, 3)   # 计算整体均值的定位误差
    LLSE_sum = LLSE_sum_list.mean(0)
    error_sum = distance(LLSE_sum, tagposition)

    LM_RMSE = error_list.mean(0)                     # 计算两算法整体定位误差均值
    LLSE_RMSE = error_LLSE_list.mean(0)
    LLSE_sub_RMSE_1 = error_LLSE_sub_list_1.mean(0)
    LLSE_sub_RMSE_2 = error_LLSE_sub_list_2.mean(0)


    LLSE_height_RMSE = abs(LLSE_height_list - tagposition[2])   # 计算两算法高度误差的均值
    LLSE_height_RMSE = LLSE_height_RMSE.mean(0)
    LM_height_RMSE = abs(LM_height_list - tagposition[2])
    LM_height_RMSE = LM_height_RMSE.mean(0)

    LLSE_Mean_Height = LLSE_height_list.mean(0)     # 计算两算法的高度结果均值
    LM_Mean_Height = LM_height_list.mean(0)


    print('\n' + 'LSMR法整体均方根误差: ' + str(LM_RMSE) + "米")
    print('LLSE法整体均方根误差: ' + str(LLSE_RMSE) + "米")
    print('LLSE法子集0均方根误差: ' + str(LLSE_sub_RMSE_1) + "米")
    print('LLSE法各子集加权均方根误差: ' + str(LLSE_sub_RMSE_2) + "米")
    print('LLSE法' + str(test_number) + '次求平均误差: ' + str(error_sum) + "米")

    print('\n' + 'LLSE法粗估计高度均方根误差: ' + str(LLSE_height_RMSE) + "米")
    print('LLSE法粗估计求得平均高度: ' + str(LLSE_Mean_Height) + "米")
    print('\n')
    print('\n' + 'LSMR法最终高度均方根误差: ' + str(LM_height_RMSE) + "米")
    print('LSMR法最终求得平均高度: ' + str(LM_Mean_Height) + "米")

    x = np.arange(0, LLSE_height_list.shape[0])
    plt.xlabel('Measurement number : ' + str(LLSE_height_list.shape[0]))
    plt.title('The Z axis error of LLSE')
    plt.plot(x, LLSE_height_list - tagposition[2], 'ob', markersize=2)
    plt.show()

    x = np.arange(0, LM_height_list.shape[0])
    plt.xlabel('Measurement number : ' + str(LM_height_list.shape[0]))
    plt.title('The Z axis error of LSMR')
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

