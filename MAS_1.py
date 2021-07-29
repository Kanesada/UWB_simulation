'''code = 'utf-8'''

'''author = peng'''

import copy

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import animation

import time

NUM = 10  # 设置无人机数量

MOVE_DISTANCE = 0.3

JianCeError = 0.1

'''根据无人机数量NUM得出边界最大容量数量   :    MAXNUM'''

if ((NUM - 4) / 4) % 1 == 0:

    MAXNUM = (NUM - 4) / 4

else:
    MAXNUM = int((NUM - 4) / 4) + 1

'''JIANJU是调整单位距离'''

JIANJU = 50 / (MAXNUM + 1)

x = np.random.randint(1, 100, NUM)

y = np.random.randint(1, 100, NUM)

# x = [36,37,38,39]

# y = [36,37,38,39]

Point_list = []

for i in range(NUM):
    Point_list.append([x[i], y[i]])

DING_LIST = [[25, 25], [75, 25], [75, 75], [25, 75]]

DingX, DingY = [], []

for each in DING_LIST:
    DingX.append(each[0])

    DingY.append(each[1])

DingX.append(DING_LIST[0][0])

DingY.append(DING_LIST[0][1])

fig, ax = plt.subplots()

ax.set_xlim(0, 100)

ax.set_ylim(0, 100)

sc = ax.scatter(x, y, color='r', alpha=0.7, marker='1', linewidth=10)

ax.plot(DingX, DingY, color='black', linestyle=':')


class Point():
    MOVE_DISTANCE = MOVE_DISTANCE

    JianCeError = JianCeError

    MAXNUM = MAXNUM

    JIANJU = JIANJU

    tiaozheng_aim = None

    def __init__(self, id):

        self.id = id

    def decide(self, list=copy.deepcopy(DING_LIST)):

        if self.tiaozheng_aim == None:  # 调整目标定下来就不需要改变了

            nearest = self.detect_nearest(list)  # 检测最近顶点

            ID = self.occupy(nearest)  # 检测占领者

            if ID == self.id:

                self.update(nearest)

                pass  # 自己占领

            elif ID == None:
                self.update(nearest)  # 无人占领，往该方向移动

            else:  # self.update([50,50])

                self.tiaozheng_aim = self.adjust(ID)  # 调整目标

                if self.tiaozheng_aim:  # 调整成功

                    self.update(self.tiaozheng_aim)

                else:  # 调整失败

                    # print(list)

                    list2 = copy.deepcopy(list)  # 深复制防出错

                    list2.remove(nearest)

                    # print(list)

                    return self.decide(list2)

        else:
            self.update(self.tiaozheng_aim)  # 有调整目标，直接移往该方向

    def adjust(self, ID):

        order = obj_list[ID].send()  # 1,0

        if order == None: return None

        for each in DING_LIST:

            d = self.distance_calculate(each, Point_list[ID])

            if d < self.JianCeError:
                identity = DING_LIST.index(each)

        aim = copy.deepcopy(DING_LIST[identity])

        count = self.MAXNUM - order  # 1,2

        if count % 2 == 0:  # 偶数顺时针

            if identity == 3:

                aim[0] += self.JIANJU * (count / 2)

                return aim

            elif identity == 2:

                aim[1] -= self.JIANJU * (count / 2)

                return aim

            elif identity == 1:

                aim[0] -= self.JIANJU * (count / 2)

                return aim

            else:

                aim[1] += self.JIANJU * (count / 2)

                return aim

        elif identity == 3:  # 奇数逆时针

            aim[1] -= self.JIANJU * (int((count / 2)) + 1)

            return aim

        elif identity == 2:

            aim[0] -= self.JIANJU * (int((count / 2)) + 1)

            return aim

        elif identity == 1:

            aim[1] += self.JIANJU * (int((count / 2)) + 1)

            return aim

        else:

            aim[0] += self.JIANJU * (int((count / 2)) + 1)

            return aim

    def detect_nearest(self, list):

        init_distance = self.distance_calculate(Point_list[self.id], list[0])

        count, i = 0, 0

        for each in list:

            D = self.distance_calculate(Point_list[self.id], each)

            if D < init_distance:
                init_distance = D

                count = i

            i += 1

        return list[count]

    def distance_calculate(self, A, B):  # [1,1],[2,2] 得1.4142135623730951

        return pow(pow(abs(A[0] - B[0]), 2) + pow(abs(A[1] - B[1]), 2), 0.5)

    def update(self, aim):

        self_pot = copy.deepcopy(Point_list[self.id])

        x = np.array([aim[0] - self_pot[0], aim[1] - self_pot[1]])  # 方向向量

        y = np.array([1, 0])  # x轴方向

        Lx = np.sqrt(x.dot(x))  # x.dot(x) 点乘自己，相当于向量模平方

        Ly = np.sqrt(y.dot(y))

        if Lx > self.MOVE_DISTANCE:

            cos_angle = x.dot(y) / (Lx * Ly)

            angle = np.arccos(cos_angle)  # 0.....pi

            if x[0] >= 0 and x[1] >= 0:

                self_pot[0] = self_pot[0] + self.MOVE_DISTANCE * abs(np.cos(angle))

                self_pot[1] = self_pot[1] + self.MOVE_DISTANCE * np.sin(angle)

            elif x[0] <= 0 and x[1] >= 0:

                self_pot[0] = self_pot[0] - self.MOVE_DISTANCE * abs(np.cos(angle))

                self_pot[1] = self_pot[1] + self.MOVE_DISTANCE * np.sin(angle)

            elif x[0] <= 0 and x[1] <= 0:

                self_pot[0] = self_pot[0] - self.MOVE_DISTANCE * abs(np.cos(angle))

                self_pot[1] = self_pot[1] - self.MOVE_DISTANCE * np.sin(angle)

            else:

                self_pot[0] = self_pot[0] + self.MOVE_DISTANCE * abs(np.cos(angle))

                self_pot[1] = self_pot[1] - self.MOVE_DISTANCE * np.sin(angle)

            Point_list[self.id] = self_pot

        else:

            Point_list[self.id] = aim

    def occupy(self, nearest):

        for each in Point_list:

            d = self.distance_calculate(each, nearest)

            if d < self.JianCeError:
                ID = Point_list.index(each)

                return ID

        return None

    def send(self):

        '''self.MAXNUM = 2 ,则输出 1,0'''

        if self.MAXNUM <= 0:

            return None  # 告诉询问着索要失败

        else:

            self.MAXNUM -= 1

            return self.MAXNUM


obj_list = [Point(i) for i in range(0, NUM)]  # 返回生成的NUM个对象的列表

comp_li = None


def gen():  # 绘图函数里面用的数据来源

    global comp_li

    while True:

        li = []

        for i in range(NUM):
            obj_list[i].decide()

        for each in Point_list:
            li.append(each)

        if comp_li == li:

            print('抵达边界完成，停留3秒')

            time.sleep(3)

            exit()

        else:
            comp_li = copy.deepcopy(li)

        with open('set.py', 'w') as f:

            f.write('POINT_LIST = ' + str(li))

        yield li


def update(N_list):
    sx, sy = [], []

    for each in N_list:
        sx.append(each[0])

        sy.append(each[1])

        sc.set_offsets(np.c_[sx, sy])

    return sc


ani = animation.FuncAnimation(fig, update, frames=gen, interval=1)

plt.show()
