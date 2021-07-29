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
#狭长布局 X方向长约20米 Y方向宽约2米 基站高度方向最大落差0.32米
#S = np.array([[0.2, 10.6, 2.84], [16.0, 12.648, 3], [0.5, 12.648, 3.08], [19.231, 10.6, 3.16]])
S = np.array([[0.2, 10.6, 2.84], [3.354, 11.432, 3.08], [7.816, 10.8, 3.16]])  #会议室基站布局
S_x = S[:, 0] # 各基站X值
S_y = S[:, 1] # 各基站Y值
S_z = S[:, 2] # 各基站Z值

bound_min = np.array([min(S_x), min(S_y), -0.01])
bound_max = np.array([max(S_x), max(S_y), max(S_z)])
Bound = np.array([bound_min, bound_max])
#print(Bound)

numAnchors = S.shape[0]

######## 在此处设置标签坐标（x ，y ，z） ##########
tagposition = np.array([2.5, 11.3, 1.2])

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
    # result_robust = least_squares(f, Guess, method='trf', bounds=Bound,
    #                               tr_options={'regularize': 1}, tr_solver='lsmr')['x']

    result_robust = least_squares(f, Guess, method='lm')['x']
    return result_robust



# 绘制三维可视化示意图
def Draw3D():
    ax = plt.axes(projection='3d')
    ax.scatter3D(tagposition[0], tagposition[1], tagposition[2], color='blue')
    for i in range(numAnchors):
        ax.scatter3D(S[i, 0], S[i, 1], S[i, 2],  color='red')
    #ax.scatter3D(result[0], result[1], result[2], color='green')
    #ax.scatter3D(result2[0], result2[1], result2[2], color='purple')
    ax.scatter3D(LLSE_sum[0], LLSE_sum[1], LLSE_sum[2], color='black')
    plt.xlabel('Red:Anchor  Blue:True tag Black:(Averaged)Result')
    plt.show()

# 统计100次测量
def One_hundred():
    test_number = 10
    global error_list, result2, LM_height_list, LLSE_height_list, error_LLSE_list, result, LLSE_sum
    LLSE_sum_list = np.array([])
    LSMR_width_list = np.array([])
    LSMR_length_list = np.array([])

    for i in range(test_number):    # 进行100次测量
        Generate_LOS_doa()
        result = LLSE()
        result2 = Get_result(bound_min + 0.5*(bound_max-bound_min))
        if i <= test_number:
            LLSE_sum_list = np.append(LLSE_sum_list, result2)


        error = distance(result2, tagposition)             # 计算两算法整体定位误差
        error_LLSE = distance(result, tagposition)


        error_list = np.append(error_list, error)         # 生成两算法整体定位误差列表
        error_LLSE_list = np.append(error_LLSE_list, error_LLSE)


        LLSE_height_list = np.append(LLSE_height_list, result[2])      # 生成两算法高度结果列表
        LM_height_list = np.append(LM_height_list, result2[2])

        LSMR_width_list = np.append(LSMR_width_list, result2[1])      # 生成LSMR横向坐标列表
        LSMR_length_list = np.append(LSMR_length_list, result2[0])      # 生成LSMR纵向坐标列表



    LLSE_sum_list = LLSE_sum_list.reshape(-1, 3)   # 计算整体均值的定位误差
    LLSE_sum = LLSE_sum_list.mean(0)
    error_sum = distance(LLSE_sum, tagposition)

    LM_RMSE = error_list.mean(0)                     # 计算两算法整体定位误差均值
    LLSE_RMSE = error_LLSE_list.mean(0)



    LSMR_length_RMSE = abs(LSMR_length_list - tagposition[0])   # 计算纵向误差均值
    LSMR_length_RMSE = LSMR_length_RMSE.mean(0)

    LSMR_width_RMSE = abs(LSMR_width_list - tagposition[1])   # 计算横向误差均值
    LSMR_width_RMSE = LSMR_width_RMSE.mean(0)

    LM_height_RMSE = abs(LM_height_list - tagposition[2])     # 计算高度误差均值
    LM_height_RMSE = LM_height_RMSE.mean(0)


    LSMR_Mean_length = LSMR_length_list.mean(0)  # 计算三轴坐标均值
    LSMR_Mean_Width = LSMR_width_list.mean(0)
    LM_Mean_Height = LM_height_list.mean(0)

    print('LLSE法整体均方根误差: ' + str(LLSE_RMSE) + "米")
    print('\n' + 'LSMR法整体均方根误差: ' + str(LM_RMSE) + "米")
    print('LSMR法' + str(test_number) + '次求平均误差: ' + str(error_sum) + "米")
    print('\n')

    print('\n' + 'LSMR法最终X轴均方根误差: ' + str(LSMR_length_RMSE) + "米")
    print('LSMR法最终求得平均X: ' + str(LSMR_Mean_length) + "米")

    print('\n' + 'LSMR法最终Y轴均方根误差: ' + str(LSMR_width_RMSE) + "米")
    print('LSMR法最终求得平均Y: ' + str(LSMR_Mean_Width) + "米")

    print('\n' + 'LSMR法最终Z轴均方根误差: ' + str(LM_height_RMSE) + "米")
    print('LSMR法最终求得平均Z: ' + str(LM_Mean_Height) + "米")


    x = np.arange(0, LSMR_length_list.shape[0])
    plt.xlabel('Measurement number : ' + str(LSMR_length_list.shape[0]))
    plt.title('The X axis error of LSMR')
    plt.plot(x, LSMR_length_list - tagposition[0], 'ob', markersize=2)
    plt.show()

    x = np.arange(0, LSMR_width_list.shape[0])
    plt.xlabel('Measurement number : ' + str(LSMR_width_list.shape[0]))
    plt.title('The Y axis error of LSMR')
    plt.plot(x, LSMR_width_list - tagposition[1], 'ob', markersize=2)
    plt.show()

    x = np.arange(0, LM_height_list.shape[0])
    plt.xlabel('Measurement number : ' + str(LM_height_list.shape[0]))
    plt.title('The Z axis error of LSMR')
    plt.plot(x, LM_height_list - tagposition[2], 'ob', markersize=2)
    plt.show()

    # result2[2] = LM_Mean_Height
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

