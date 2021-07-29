'''重写 13 '''

'''基本ok 只差停下函数'''

'''哇 终于TM的停下来了'''

import copy

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import animation

import time

from set import POINT_LIST

Move_Distance = 20  # 20 * 0.01 =0.2

Ting_Distance = 3

# POINT_LIST = [[41.66666666666667, 25], [25, 41.66666666666667], [41.66666666666667, 75], [75, 25], [25, 75], [75, 58.33333333333333], [75, 75], [58.33333333333333, 75], [25, 25], [75, 41.66666666666667], [25, 58.33333333333333], [58.33333333333333, 25]]

# Point_list = [[25, 50.0], [75, 43.75], [43.75, 25], [25, 75], [25, 43.75], [75, 68.75], [56.25, 25], [62.5, 75], [50.0, 25], [75, 62.5], [25, 68.75], [31.25, 75], [25, 25], [31.25, 25], [25, 31.25], [75, 50.0], [37.5, 25], [56.25, 75], [75, 25], [75, 75], [75, 31.25], [25, 62.5], [37.5, 75], [68.75, 25], [75, 37.5], [25, 37.5], [25, 56.25], [68.75, 75], [62.5, 25], [43.75, 75]]

# Point_list = [[25, 25], [75, 75], [25, 75], [75, 25], [50, 25]]

# Point_list = [[25, 43.75], [25, 56.25], [50.0, 25], [75, 37.5], [68.75, 75], [43.75, 75], [62.5, 25], [75, 43.75], [25, 75], [25, 25], [56.25, 25], [25, 68.75], [75, 50.0], [31.25, 75], [25, 62.5], [75, 68.75], [31.25, 25], [25, 31.25], [62.5, 75], [75, 62.5], [56.25, 75], [75, 56.25], [37.5, 25], [75, 25], [75, 31.25], [25, 37.5], [68.75, 25], [37.5, 75], [43.75, 25]]

Point_list = POINT_LIST

NUM = len(Point_list)

# print(NUM)

DING_LIST = [[25, 25], [75, 25], [75, 75], [25, 75]]

DingX, DingY, x, y = [], [], [], []

for each in DING_LIST:
    DingX.append(each[0])

    DingY.append(each[1])

for each in Point_list:
    x.append(each[0])

    y.append(each[1])

DingX.append(DING_LIST[0][0])

DingY.append(DING_LIST[0][1])

fig, ax = plt.subplots()

ax.set_xlim(0, 100)

ax.set_ylim(0, 100)

sc = ax.scatter(x, y, color='r', alpha=0.7, marker='1', linewidth=10)

ax.plot(DingX, DingY, color='black', linestyle=':')

'''以间隔0.01生成齿轮链表'''


def chain_make():
    Tooth_Chain = []

    Tooth_Chain.append([25, 25])

    for i in np.arange(25.01, 75, 0.01):
        Tooth_Chain.append([i, 25])

    Tooth_Chain.append([75, 25])

    for i in np.arange(25.01, 75, 0.01):
        Tooth_Chain.append([75, i])

    Tooth_Chain.append([75, 75])

    for i in np.arange(74.99, 25.0, -0.01):
        Tooth_Chain.append([round(i, 2), 75])

    Tooth_Chain.append([25, 75])

    for i in np.arange(74.99, 25, -0.01):
        Tooth_Chain.append([25, round(i, 2)])

    return Tooth_Chain


def distance_calculate(A, B):  # [1,1],[2,2] 得1.4142135623730951

    return pow(pow(abs(A[0] - B[0]), 2) + pow(abs(A[1] - B[1]), 2), 0.5)


Tooth_Chain = chain_make()

Tooth_Len = len(Tooth_Chain)

Point_adindex = []

for a in Point_list:

    for b in Tooth_Chain:

        d = distance_calculate(a, b)

        if d <= 0.005:  # Point_list数据有问题

            a.append(Tooth_Chain.index(b))

            Point_adindex.append(a)


# print(len(Point_adindex))


def takeThird(elem):
    return elem[2]


Point_adindex_sort = copy.deepcopy(Point_adindex)

Point_adindex_sort.sort(key=takeThird)


# print(len(Point_adindex_sort))


class Point():
    next_dis = 200001

    def __init__(self, id):

        ''' self.  pre_id    next_id     id  这三个是在Point_list中的位置'''

        self.id = id

        my_id = Point_adindex_sort.index(Point_adindex[self.id])

        if my_id == 0:

            self.pre_id = Point_adindex.index(Point_adindex_sort[NUM - 1])

            self.next_id = Point_adindex.index(Point_adindex_sort[1])

        elif my_id == NUM - 1:

            self.next_id = Point_adindex.index(Point_adindex_sort[0])

            self.pre_id = Point_adindex.index(Point_adindex_sort[NUM - 2])

        else:

            self.pre_id = Point_adindex.index(Point_adindex_sort[my_id - 1])

            self.next_id = Point_adindex.index(Point_adindex_sort[my_id + 1])

    def decide(self):

        pre_chain_index = Point_adindex[self.pre_id][2]

        next_chain_index = Point_adindex[self.next_id][2]

        self_chain_index = Point_adindex[self.id][2]

        if pre_chain_index < next_chain_index:

            a = pre_chain_index

            b = next_chain_index

        else:

            a = pre_chain_index

            b = next_chain_index + 20000

        if abs(self_chain_index - (a + b) / 2) < 100:
            pass

        else:

            if pre_chain_index < next_chain_index:  # 正常情况

                self.next_dis = next_chain_index - self_chain_index

                mmid = ((next_chain_index + pre_chain_index) / 2 + self_chain_index) / 2

                # print('pre:', pre_chain_index, ' ', 'self', self_chain_index, ' ', 'next:', next_chain_index)

            else:

                self.next_dis = next_chain_index - self_chain_index + 20000

                if self.next_dis >= 20000:
                    self.next_dis -= 20000

                mmid = ((next_chain_index + Tooth_Len + pre_chain_index) / 2 + self_chain_index) / 2

                # print('pre:', pre_chain_index, ' ', 'self', self_chain_index, ' ', 'next:', next_chain_index)

            if abs(mmid - self_chain_index) <= Ting_Distance:

                if mmid % 1 == 0:

                    self.move(int(mmid))

                elif self_chain_index > mmid:  # 在目标顺市针方向

                    self.move(int(mmid) + 1)

                else:

                    self.move(int(mmid))

            elif mmid > self_chain_index:

                self.move(self_chain_index + Move_Distance)

            else:

                self.move(self_chain_index - Move_Distance)

    def move(self, aim):

        if aim >= Tooth_Len: aim -= Tooth_Len

        li = copy.deepcopy(Tooth_Chain[aim])

        li.append(aim)

        Point_adindex[self.id] = li


def judge(list):
    d = 20000 / NUM

    for each in list:

        if abs(each - d) > 100:
            return False

    return True


def gen():
    while True:

        # print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

        li = []

        # panduanls=[]

        # if vari > ? :

        for i in range(NUM):
            obj_list[i].decide()

            # panduanls.append(obj_list[i].next_dis)

        # else:continue

        # if judge(panduanls):

        #     print("均匀化分布算法执行完毕，停留3秒")

        #     time.sleep(3)

        #     exit()

        for each in Point_adindex: li.append(each[:-1])

        yield li


def update(N_list):
    sx, sy = [], []

    for each in N_list:
        sx.append(each[0])

        sy.append(each[1])

        sc.set_offsets(np.c_[sx, sy])

    return sc


obj_list = [Point(i) for i in range(0, len(Point_list))]  # 返回生成的NUM个对象的列表

ani = animation.FuncAnimation(fig, update, frames=gen, interval=2)

plt.show()

###均匀化代码需要数据POINT_LIST,可以用代码中注释掉的数据看效果。
