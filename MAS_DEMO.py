import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

NUM = 10
n = np.random.randint(3,10)  # 3、4、5边形
R = 30  # 多边形半径
Point_List = []
'''高度错开所需参数'''
low, SPEED0, Hight = 0, 2, 200
distance = 380 / NUM

'''抵达边界所需参数 , # 根据无人机数量NUM得出边界最大容量数量MAXNUM'''
MOVE_DISTANCE = 0.3
JianCeError = 0.1
if ((NUM - n) / n) % 1 == 0:
    MAXNUM = (NUM - n) / n
else:
    MAXNUM = int((NUM - n) / n) + 1
JIANJU = 2 * R * np.sin(np.pi / n) / (MAXNUM + 1)
# 2 * R * np.sin(np.pi/n)是边界长度， JIANJU是调整单位距离

'''均匀化参数'''
Ting_Distance = 3
Move_Distance = 20  # 20 * 0.01 =0.2
'''随机生成初始分布位置 , 设定多边形位置 '''
x = np.random.randint(1, 100, NUM)
y = np.random.randint(1, 100, NUM)
z = [0 for i in range(NUM)]
for i in range(NUM):
    Point_List.append([x[i], y[i], z[i]])
# 多边形数量设定
DING_LIST = []
DingX, DingY, DingZ = [], [], []


def PolygonInit():
    global DING_LIST
    global n
    global DingX, DingY, DingZ
    for i in range(1, n + 1):
        x = 50 + R * np.sin(i * 2 * np.pi / n)
        y = 50 + R * np.cos(i * 2 * np.pi / n)
        DING_LIST.append([x, y, Hight])
    DingX, DingY, DingZ = [], [], []
    for each in DING_LIST:
        DingX.append(each[0])
        DingY.append(each[1])
        DingZ.append(each[2])
    DingX.append(DING_LIST[0][0])
    DingY.append(DING_LIST[0][1])
    DingZ.append(DING_LIST[0][2])


PolygonInit()  # 初始化多边形顶点位置

'''初始化图像'''
fig = plt.figure()
ax = plt.gca(projection='3d')
ax.set_xlim(0, 100)
ax.set_xlabel('X')
ax.set_ylim(0, 100)
ax.set_ylabel('Y')
ax.set_zlim(0, 400)
ax.set_zlabel('Z')
sc = ax.scatter3D(x, y, z, color='r', alpha=0.7)
# ax.plot(DingX, DingY, DingZ, 'b:')
#sc = ax.scatter3D(x, y,z, color='r', alpha=0.7,marker='1',linewidth = 8)
ax.plot(DingX, DingY,DingZ, color = 'black',linestyle = ':')

D = 2 * R * np.sin(np.pi / n)
num = int(D / 0.01)
tooth_distance = D / num
Tooth_Chain = []


def Chain_make():
    global Tooth_Chain
    for i in range(0, n):
        Tooth_Chain.append(DING_LIST[i])
        base_pos = copy.deepcopy(DING_LIST[i])
        if i == len(DING_LIST) - 1:
            next = DING_LIST[0]
        else:
            next = DING_LIST[i + 1]
        x = np.array([next[0] - DING_LIST[i][0], next[1] - DING_LIST[i][1]])  # 方向向量
        y = np.array([1, 0])  # x轴方向
        Lx = np.sqrt(x.dot(x))  # x.dot(x) 点乘自己，相当于向量模平方
        Ly = np.sqrt(y.dot(y))
        cos_angle = x.dot(y) / (Lx * Ly)
        angle = np.arccos(cos_angle)
        if x[0] >= 0 and x[1] >= 0:
            for j in range(1, num):
                a = base_pos[0] + j * tooth_distance * abs(np.cos(angle))
                b = base_pos[1] + j * tooth_distance * abs(np.sin(angle))
                Tooth_Chain.append([a, b, Hight])
            # print('1', len(Tooth_Chain))
        elif x[0] <= 0 and x[1] >= 0:
            for j in range(1, num):
                a = base_pos[0] - j * tooth_distance * abs(np.cos(angle))
                b = base_pos[1] + j * tooth_distance * abs(np.sin(angle))
                Tooth_Chain.append([a, b, Hight])
            # print('2', len(Tooth_Chain))
        elif x[0] <= 0 and x[1] <= 0:
            for j in range(1, num):
                a = base_pos[0] - j * tooth_distance * abs(np.cos(angle))
                b = base_pos[1] - j * tooth_distance * abs(np.sin(angle))
                Tooth_Chain.append([a, b, Hight])
            # print('3', len(Tooth_Chain))
        else:
            for j in range(1, num):
                a = base_pos[0] + j * tooth_distance * abs(np.cos(angle))
                b = base_pos[1] - j * tooth_distance * abs(np.sin(angle))
                Tooth_Chain.append([a, b, Hight])
            # print('4', len(Tooth_Chain))
    return Tooth_Chain


def distance_calculate_2D(A, B):
    return pow(pow(A[0] - B[0], 2) + pow(A[1] - B[1], 2), 0.5)


Chain_make()
Tooth_Len = len(Tooth_Chain)


class Point():
    def __init__(self, id):
        self.id = id
        self.random = np.random.random()
        self.aim_hight = None
        self.tiaozheng_aim = None
        self.MAXNUM = MAXNUM

    def decide0(self):
        my_pos = copy.deepcopy(Point_List[self.id])
        if abs(Hight - my_pos[2]) < SPEED0:
            Point_List[self.id][2] = Hight
        else:
            aim = my_pos[2] + SPEED0
            Point_List[self.id][2] = aim

    def decide1(self):
        if self.aim_hight == None:
            li1 = []
            for i in range(NUM):
                ran = obj_list[i].send_random()
                li1.append([ran, i])

                def takeFirst(elem):
                    return elem[0]

                li1.sort(key=takeFirst)
            my_index = li1.index([self.random, self.id])
            self.aim_hight = 10 + (my_index + 1) * distance  # 因为my_index从0 开始取
        # 有目标高度后开始更新位置
        my_pos = copy.deepcopy(Point_List[self.id])
        if abs(my_pos[2] - self.aim_hight) < SPEED0:
            Point_List[self.id][2] = self.aim_hight
        elif my_pos[2] < self.aim_hight:
            Point_List[self.id][2] = my_pos[2] + SPEED0
        else:
            Point_List[self.id][2] = my_pos[2] - SPEED0

    def decide2(self, list=copy.deepcopy(DING_LIST)):
        if self.tiaozheng_aim == None:
            nearest = self.detect_nearest(list)  # 检测最近顶点
            ID = self.occupy(nearest)
            if ID == self.id:
                self.update(nearest)
            elif ID == None:
                self.update(nearest)
            else:
                self.tiaozheng_aim = self.adjust(ID)
                if self.tiaozheng_aim:  # 调整成功
                    self.update(self.tiaozheng_aim)
                else:
                    list2 = copy.deepcopy(list)
                    list2.remove(nearest)
                    return self.decide2(list2)
        else:
            self.update(self.tiaozheng_aim)

    def decide3(self):
        pass

    def decide4(self):
        pass

    def send_random(self):
        return self.random

    def detect_nearest(self, list):
        init_distance = self.distance_calculate(Point_List[self.id], list[0])
        count, i = 0, 0
        for each in list:
            D = self.distance_calculate(Point_List[self.id], each)
            if D < init_distance:
                init_distance = D
                count = i
            i += 1
        return list[count]

    def distance_calculate(self, A, B):  # [1,1,?],[2,2,?] 得1.4142135623730951
        return pow(pow(abs(A[0] - B[0]), 2) + pow(abs(A[1] - B[1]), 2), 0.5)

    def occupy(self, nearest):
        for each in Point_List:
            d = self.distance_calculate(each, nearest)
            if d < JianCeError:
                ID = Point_List.index(each)
                return ID
        return None

    def update(self, aim):
        self_pot = copy.deepcopy(Point_List[self.id])
        x = np.array([aim[0] - self_pot[0], aim[1] - self_pot[1]])  # 方向向量
        y = np.array([1, 0])  # x轴方向
        Lx = np.sqrt(x.dot(x))  # x.dot(x) 点乘自己，相当于向量模平方
        Ly = np.sqrt(y.dot(y))
        if Lx > MOVE_DISTANCE:
            cos_angle = x.dot(y) / (Lx * Ly)
            angle = np.arccos(cos_angle)  # 0.....pi
            if x[0] >= 0 and x[1] >= 0:
                self_pot[0] = self_pot[0] + MOVE_DISTANCE * abs(np.cos(angle))
                self_pot[1] = self_pot[1] + MOVE_DISTANCE * np.sin(angle)
            elif x[0] <= 0 and x[1] >= 0:
                self_pot[0] = self_pot[0] - MOVE_DISTANCE * abs(np.cos(angle))
                self_pot[1] = self_pot[1] + MOVE_DISTANCE * np.sin(angle)
            elif x[0] <= 0 and x[1] <= 0:
                self_pot[0] = self_pot[0] - MOVE_DISTANCE * abs(np.cos(angle))
                self_pot[1] = self_pot[1] - MOVE_DISTANCE * np.sin(angle)
            else:
                self_pot[0] = self_pot[0] + MOVE_DISTANCE * abs(np.cos(angle))
                self_pot[1] = self_pot[1] - MOVE_DISTANCE * np.sin(angle)
            Point_List[self.id] = self_pot
        else:
            Point_List[self.id][0] = aim[0]
            Point_List[self.id][1] = aim[1]

    def adjust(self, ID):
        order = obj_list[ID].send_order()
        if order == None: return None
        for each in DING_LIST:
            d = self.distance_calculate(each, Point_List[ID])
            if d < JianCeError:
                identity = DING_LIST.index(each)
        aim = copy.deepcopy(DING_LIST[identity])
        count = self.MAXNUM - order  # 1,2
        if identity == 0:
            pre = DING_LIST[-1]
            next = DING_LIST[identity + 1]
        elif identity == len(DING_LIST) - 1:
            pre = DING_LIST[identity - 1]
            next = DING_LIST[0]
        else:
            pre = DING_LIST[identity - 1]
            next = DING_LIST[identity + 1]

        if count % 2 == 0:  # 偶数顺时针
            x = np.array([pre[0] - aim[0], pre[1] - aim[1]])  # 方向向量
        else:  # 奇数逆时针
            x = np.array([next[0] - aim[0], next[1] - aim[1]])  # 方向向量
        count2 = count / 2 if count % 2 == 0 else int(count / 2) + 1
        y = np.array([1, 0])  # x轴方向
        Lx = np.sqrt(x.dot(x))  # x.dot(x) 点乘自己，相当于向量模平方
        Ly = np.sqrt(y.dot(y))
        cos_angle = x.dot(y) / (Lx * Ly)
        angle = np.arccos(cos_angle)
        if x[0] >= 0 and x[1] >= 0:

            aim[0] = aim[0] + count2 * JIANJU * abs(np.cos(angle))
            aim[1] = aim[1] + count2 * JIANJU * np.sin(angle)
        elif x[0] <= 0 and x[1] >= 0:
            aim[0] = aim[0] - count2 * JIANJU * abs(np.cos(angle))
            aim[1] = aim[1] + count2 * JIANJU * np.sin(angle)
        elif x[0] <= 0 and x[1] <= 0:
            aim[0] = aim[0] - count2 * JIANJU * abs(np.cos(angle))
            aim[1] = aim[1] - count2 * JIANJU * np.sin(angle)
        else:
            aim[0] = aim[0] + count2 * JIANJU * abs(np.cos(angle))
            aim[1] = aim[1] - count2 * JIANJU * np.sin(angle)
        return aim

    def send_order(self):
        if self.MAXNUM <= 0:
            return None  # 告诉询问着索要失败
        else:
            self.MAXNUM -= 1
            return self.MAXNUM


class Point2():
    next_dis = Tooth_Len + 1
    def __init__(self, id):
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

    def decide1(self):
        pre_chain_index = Point_adindex[self.pre_id][3]
        next_chain_index = Point_adindex[self.next_id][3]
        self_chain_index = Point_adindex[self.id][3]
        if pre_chain_index < next_chain_index:  # 正常情况
            self.next_dis = next_chain_index - self_chain_index
            mmid = ((next_chain_index + pre_chain_index) / 2 + self_chain_index) / 2
        else:
            self.next_dis = next_chain_index - self_chain_index + Tooth_Len
            if self.next_dis >= Tooth_Len:
                self.next_dis -= Tooth_Len
            mmid = ((next_chain_index + Tooth_Len + pre_chain_index) / 2 + self_chain_index) / 2

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

    def decide2(self):
        my_hight = Point_List[self.id][2]
        if abs(my_hight - Hight) < SPEED0:
            Point_List[self.id][2] = Hight
        elif my_hight > Hight:
            Point_List[self.id][2] = my_hight - SPEED0
        else:
            Point_List[self.id][2] = my_hight + SPEED0




    def move(self, aim):
        if aim >= Tooth_Len: aim -= Tooth_Len
        li = copy.deepcopy(Tooth_Chain[aim])
        Point_List[self.id][0] = copy.deepcopy(Tooth_Chain[aim])[0]
        Point_List[self.id][1] = copy.deepcopy(Tooth_Chain[aim])[1]
        li.append(aim)
        Point_adindex[self.id] = li


obj_list = [Point(i) for i in range(0, NUM)]  # 返回生成的NUM个对象的列表
# obj_list2 = [Point2(i) for i in range(0, NUM)]  # 返回生成的NUM个对象的列表

Point_adindex_sort = []


def gen():
    global Point_adindex_sort
    global Point_adindex
    global comp_li
    global Tooth_Chain
    state = 0
    Point_list2 = []  # 用于比较
    init = 0
    while True:
        panduanls = []
        # print(Point_List)
        if state == 0:
            for i in range(NUM):
                obj_list[i].decide0()
        elif state == 1:
            for i in range(NUM):
                obj_list[i].decide1()
        elif state == 2:
            for i in range(NUM):
                obj_list[i].decide2()
        elif state == 3:

            if init == 0:
                Point_adindex = []
                for i in Point_List:
                    for j in Tooth_Chain:
                        if distance_calculate_2D(i, j) <= tooth_distance / 2:
                            li = copy.deepcopy(i)
                            li.append(Tooth_Chain.index(j))
                            Point_adindex.append(li)
                            break

                def takeThird(elem):
                    return elem[3]

                Point_adindex_sort = copy.deepcopy(Point_adindex)
                Point_adindex_sort.sort(key=takeThird)
                obj_list2 = [Point2(i) for i in range(0, NUM)]  # 返回生成的NUM个对象的列表
                init = 1

            for i in range(NUM):
                obj_list2[i].decide1()
                panduanls.append(obj_list2[i].next_dis)
            if judge(panduanls):
                state += 1
        elif state == 4:
            for i in range(NUM):
                obj_list2[i].decide2()

        else:
            print('最终编队完成')
            time.sleep(10)
            exit()
        if Point_list2 == Point_List:
            state += 1
        else:
            pass
        Point_list2 = copy.deepcopy(Point_List)
        yield Point_List

def judge(list):
    d = Tooth_Len/NUM
    for each in list :
        if abs(each - d) > 100:
            return False
    return True

def update(N_list):
    li = np.array(N_list)
    sx, sy, sz = [], [], []
    for each in N_list:
        sx.append(each[0])
        sy.append(each[1])
        sz.append(each[2])
    sc.set_offsets(li[:, :-1])
    sc.set_3d_properties(sz, zdir='z')
    return sc


ani = animation.FuncAnimation(fig, update, frames=gen, interval=1)
plt.show()



