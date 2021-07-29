import matplotlib.pyplot as plt
import numpy as np
from pylab import *  # 让绘图显示中文的库
import sys

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 让绘图显示中文的命令


def caculate1(x1, x3, y3):
    """计算特殊情况下的直线对称点，输入的两点坐标X相同，即关于平行于Y轴的直线的对称点"""
    x4 = 2 * x1 - x3
    y4 = y3
    plt.scatter([x4], [y4], color='blue', marker='*', label='所求对称点')
    plt.legend()
    plt.show()


def caculate2(x1, x3, y3):
    """计算特殊情况下的直线对称点，输入的两点坐标Y相同，即关于平行于X轴的直线的对称点"""
    x4 = x3
    y4 = 2 * y1 - y3
    plt.scatter([x4], [y4], color='blue', marker='*', label='所求对称点')
    plt.legend()
    plt.show()


def caculate3(A, B, C, x3, y3):
    """计算一般情况的直线对称点，根据斜率关系推导的数学关系式"""
    x4 = x3 - 2 * A * ((A * x3 + B * y3 + C) / (A * A + B * B))
    y4 = y3 - 2 * B * ((A * x3 + B * y3 + C) / (A * A + B * B))
    plt.scatter([x4], [y4], color='blue', marker='*', label='所求对称点')
    plt.legend()
    plt.show()


while True:
    start = eval(input("\n请选择直线方程形式(1 or 2)\n1 一般式\n2 两点式\n\n"))

    # 一般式还不能画特殊情况直线

    if start == 3:
        break
        sys.exit()

    if start == 1:

        plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        plt.axis('scaled')  # 注意

        A, B, C = eval(input("请输入一般式方程三系数A,B,C(以逗号分隔):"))
        h = np.linspace(0, 8, 10)
        z = (-A * h - C) / B
        plt.plot(h, z, color='red', linewidth=2)

        x3, y3 = eval(input("\n请输入已知任意对称点X,Y坐标:"))

        # plt.text(x3,y3,str((x3,y3)))

        plt.scatter([x3], [y3], color='red', marker='+')
        caculate3(A, B, C, x3, y3)

    # 两点式支持所有直线

    elif start == 2:
        plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(" 找对称点")
        plt.axis('scaled')  # 注意

        x1, y1 = eval(input("\n请输入X1,Y1坐标(以逗号分隔下同):"))
        x2, y2 = eval(input("请输入X2,Y2坐标:"))

        x3, y3 = eval(input("\n请输入已知任意对称点X,Y坐标:"))

        if x1 == x2:
            plt.plot([x1, x2], [y1, y2], color='red', linewidth=2)
            plt.scatter([x3], [y3], color='red', marker='+', label='已知点')
            plt.legend()
            caculate1(x1, x3, y3)
        elif y1 == y2:
            plt.plot([x1, x2], [y1, y2], color='red', linewidth=2)
            plt.scatter([x3], [y3], color='red', marker='+', label='已知点')
            plt.legend()
            caculate2(x1, x3, y3)
        else:
            plt.plot([x1, x2], [y1, y2], color='red', linewidth=2)
            plt.scatter([x3], [y3], color='red', marker='+', label='已知点')
            plt.legend()
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - y1 * x2
            caculate3(A, B, C, x3, y3)
    else:
        print("请选择输入1或者2")

