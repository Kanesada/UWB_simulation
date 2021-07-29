from math import *
import numpy as np
from numpy.linalg import *
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.pyplot as plt


def chan_location(ri_1, X, Q):
    n = len(X)
    k = (X ** 2).sum(1)  # 将数组各原始平方后按列求和
    h = 0.5 * (ri_1 ** 2 - k[1:n] + k[0])
    Ga = []
    for i in range(1, n):
        Ga.append([X[i][0] - X[0][0], X[i][1] - X[0][1], ri_1[i - 1]])
    Ga = np.array(Ga)
    Ga = -Ga

    # 第一次WLS估计结果（远距算法）
    Za = inv((Ga.T).dot(inv(Q)).dot(Ga)).dot((Ga.T).dot(inv(Q)).dot(h))

    # 第一次WLS计算（近距算法）
    r = np.sqrt(((X[1:n] - Za[0:2]) ** 2).sum(1))
    B = np.diag(r)
    Fa = B.dot(Q).dot(B)
    Za1 = inv((Ga.T).dot(inv(Fa)).dot(Ga)).dot((Ga.T)).dot(inv(Fa)).dot(h)
    Za_cov = inv((Ga.T).dot(inv(Fa)).dot(Ga))

    # 第二次WLS计算（近距算法）
    Ga1 = np.array([[1, 0], [0, 1], [1, 1]])
    h1 = np.array([(Za1[0] - X[0][0]) ** 2, (Za1[1] - X[0][1]) ** 2, Za1[2] ** 2])
    B1 = np.diag([Za1[0] - X[0][0], Za1[1] - X[0][1], Za1[2]])
    Fa1 = 4 * (B1).dot(Za_cov).dot(B1)
    Za2 = inv((Ga1.T).dot(inv(Fa1)).dot(Ga1)).dot((Ga1.T)).dot(inv(Fa1)).dot(h1)

    pos1 = np.sqrt(Za2) + X[0];
    pos2 = -np.sqrt(Za2) + X[0];
    pos3 = [np.sqrt(Za2[0]), -np.sqrt(Za2[1])] + X[0]
    pos4 = [-np.sqrt(Za2[0]), np.sqrt(Za2[1])] + X[0]
    pos = [pos1, pos2]  # , pos3, pos4]
    return pos


def drawPtTest(pos, tag, X):
    figsize = 5, 4  # 设定整张图片大小
    plt.subplots(figsize=figsize)
    ax1 = plt.subplot(1, 1, 1)

    pos = np.array(pos)
    X = np.array(X)
    ax1.scatter(pos[:, 0], pos[:, 1], s=80, c='g', marker='o')
    ax1.scatter(tag[0], tag[1], s=120, c='r', marker='*')
    ax1.scatter(X[:, 0], X[:, 1], s=150, c='k', marker='1')
    plt.show()


def test():
    delta = 2
    dis = 200
    R = 200
    X = []  # 已知点坐标
    X1 = (dis, dis)
    X.append(X1)
    T = (dis * 2, dis * 2)  # 未知点，待求
    n = 7  # 已知点的个数
    r1 = sqrt((X1[0] - T[0]) ** 2 + (X1[1] - T[1]) ** 2)  # X1与T的距离
    X.append((dis + R * sqrt(3), dis))
    X.append((dis - R * sqrt(3), dis))
    X.append((dis + R * sqrt(3) / 2, dis + R * 1.5))
    X.append((dis - R * sqrt(3) / 2, dis + R * 1.5))
    X.append((dis - R * sqrt(3) / 2, dis - R * 1.5))
    X.append((dis + R * sqrt(3) / 2, dis - R * 1.5))
    X = map(list, X)  # 整体映射功能，将列表或元组转换为每个元素都为列表样式的列表
    X = np.array(X)
    Q = (0.5 * np.eye(n - 1) + 0.5 * np.ones((n - 1, n - 1))) * (delta ** 2)  # 噪声协方差矩阵
    Nerror = np.random.normal(0, delta, n - 1)  # 产生随机误差
    ri_1 = []
    for i in range(1, n):  # 各已知点与未知点之间的距离，已知
        ri_1.append(sqrt((X[i][0] - T[0]) ** 2 + (X[i][1] - T[1]) ** 2) - r1 + Nerror[i - 1])
    ri_1 = np.array(ri_1)
    pos = chan_location(ri_1, X, Q)  # 最终对比，从中选出一个正确的定位点
    drawPtTest(pos, T, X)

    # 打印结果
    print(pos)

    for i in range(0, 2):
        print(sqrt((pos[i][0] - T[0]) ** 2 + (pos[i][1] - T[1]) ** 2))


if __name__ == "__main__":
    test()
