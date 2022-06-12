# -*- coding: utf-8 -*-
import math
import numpy as np
import random

'''
PRDA方法，在不同系统负载率下，总体平均响应时间的变化
'''

# 初始化参数
# 边缘服务器节点数量
ESN_NUM = 10
# 负载率（初始）
THETA = 0.1
# 收敛条件
XI = 0.001
# 该区域内的带宽（HZ）
bandwidth = 20 * math.pow(10, 6)
# ESN-i和ESN-j的距离
esn_distance_graph_list = [[[0, 510, 1144, 1330, 1155, 1014, 532, 664, 944, 324],
                            [510, 0, 639, 922, 843, 923, 759, 470, 473, 342],
                            [1144, 639, 0, 542, 701, 1072, 1241, 755, 267, 901],
                            [1330, 922, 542, 0, 308, 774, 1195, 712, 474, 1016],
                            [1155, 843, 701, 308, 0, 467, 928, 495, 511, 832],
                            [1014, 923, 1072, 774, 467, 0, 601, 453, 822, 729],
                            [532, 759, 1241, 1195, 928, 601, 0, 506, 981, 420],
                            [664, 470, 755, 712, 495, 453, 506, 0, 488, 341],
                            [944, 473, 267, 474, 511, 822, 981, 488, 0, 668],
                            [324, 342, 901, 1016, 832, 729, 420, 341, 668, 0]],
                           [[0, 330, 974, 802, 1184, 937, 573, 421, 1185, 778],
                            [330, 0, 652, 472, 946, 820, 648, 316, 874, 536],
                            [974, 652, 0, 259, 911, 1100, 1197, 835, 551, 688],
                            [802, 472, 259, 0, 715, 846, 947, 592, 458, 435],
                            [1184, 946, 911, 715, 0, 465, 919, 772, 439, 412],
                            [937, 820, 1100, 846, 465, 0, 505, 531, 809, 412],
                            [573, 648, 1197, 947, 919, 505, 0, 365, 1132, 635],
                            [421, 316, 835, 592, 772, 531, 365, 0, 853, 383],
                            [1185, 874, 551, 458, 439, 809, 1132, 853, 0, 498],
                            [778, 536, 688, 435, 412, 412, 635, 383, 498, 0]],
                           [[0, 329, 446, 1108, 1301, 1073, 797, 659, 903, 349],
                            [329, 0, 751, 1361, 1470, 1126, 965, 639, 670, 366],
                            [446, 751, 0, 691, 976, 934, 537, 721, 1156, 544],
                            [1108, 1361, 691, 0, 461, 822, 500, 1001, 1522, 1040],
                            [1301, 1470, 976, 461, 0, 544, 509, 934, 1417, 1107],
                            [1073, 1126, 934, 822, 544, 0, 426, 497, 903, 772],
                            [797, 965, 537, 500, 509, 426, 0, 503, 1024, 607],
                            [659, 639, 721, 1001, 934, 497, 503, 0, 521, 315],
                            [903, 670, 1156, 1522, 1417, 903, 1024, 521, 0, 618],
                            [349, 366, 544, 1040, 1107, 772, 607, 315, 618, 0]],
                           [[0, 510, 1144, 1355, 1014, 532, 324, 760, 675, 1174],
                            [510, 0, 639, 945, 923, 759, 342, 326, 588, 829],
                            [1144, 639, 0, 553, 1072, 1241, 901, 443, 898, 633],
                            [1355, 945, 553, 0, 793, 1219, 1041, 622, 823, 252],
                            [1014, 923, 1072, 793, 0, 601, 729, 727, 356, 544],
                            [532, 759, 1241, 1219, 601, 0, 420, 799, 396, 980],
                            [324, 342, 901, 1041, 729, 420, 0, 476, 375, 851],
                            [760, 326, 443, 622, 727, 799, 476, 0, 476, 510],
                            [675, 588, 898, 823, 356, 396, 375, 476, 0, 588],
                            [1174, 829, 633, 252, 544, 980, 851, 510, 588, 0]],
                           [[0, 457, 1113, 1277, 934, 575, 270, 691, 1123, 727],
                            [457, 0, 670, 933, 962, 766, 346, 310, 894, 575],
                            [1113, 670, 0, 502, 1160, 1200, 915, 449, 732, 742],
                            [1277, 933, 502, 0, 910, 1120, 1018, 624, 353, 629],
                            [934, 962, 1160, 910, 0, 412, 715, 822, 565, 439],
                            [575, 766, 1200, 1120, 412, 0, 435, 770, 833, 498],
                            [270, 346, 915, 1018, 715, 435, 0, 469, 853, 458],
                            [691, 310, 449, 624, 822, 770, 469, 0, 610, 384],
                            [1123, 894, 732, 353, 565, 833, 853, 610, 0, 399],
                            [727, 575, 742, 629, 439, 498, 458, 384, 399, 0]],
                           [[0, 639, 988, 1043, 923, 719, 529, 342, 583, 611],
                            [639, 0, 432, 813, 1072, 1184, 1152, 901, 344, 634],
                            [988, 432, 0, 574, 1044, 1344, 1443, 1162, 444, 677],
                            [1043, 813, 574, 0, 578, 1046, 1318, 1038, 527, 453],
                            [923, 1072, 1044, 578, 0, 551, 946, 729, 731, 439],
                            [719, 1184, 1344, 1046, 551, 0, 457, 386, 917, 675],
                            [529, 1152, 1443, 1318, 946, 457, 0, 288, 1001, 874],
                            [342, 901, 1162, 1038, 729, 386, 288, 0, 718, 589],
                            [583, 344, 444, 527, 731, 917, 1001, 718, 0, 296],
                            [611, 634, 677, 453, 439, 675, 874, 589, 296, 0]],
                           [[0, 670, 933, 922, 1026, 725, 355, 581, 649, 733],
                            [670, 0, 502, 866, 1431, 1376, 672, 1157, 330, 996],
                            [933, 502, 0, 515, 1289, 1504, 719, 1212, 286, 838],
                            [922, 866, 515, 0, 826, 1245, 579, 922, 536, 425],
                            [1026, 1431, 1289, 826, 0, 759, 780, 532, 1161, 455],
                            [725, 1376, 1504, 1245, 759, 0, 785, 324, 1256, 844],
                            [355, 672, 719, 579, 780, 785, 0, 509, 477, 402],
                            [581, 1157, 1212, 922, 532, 324, 509, 0, 985, 521],
                            [649, 330, 286, 536, 1161, 1256, 477, 985, 0, 709],
                            [733, 996, 838, 425, 455, 844, 402, 521, 709, 0]],
                           [[0, 504, 423, 818, 1072, 1184, 998, 502, 633, 898],
                            [504, 0, 840, 1028, 931, 806, 497, 348, 756, 622],
                            [423, 840, 0, 500, 1006, 1306, 1292, 652, 463, 978],
                            [818, 1028, 500, 0, 679, 1146, 1350, 705, 280, 828],
                            [1072, 931, 1006, 679, 0, 551, 979, 623, 544, 356],
                            [1184, 806, 1306, 1146, 551, 0, 568, 691, 918, 333],
                            [998, 497, 1292, 1350, 979, 568, 0, 667, 1072, 625],
                            [502, 348, 652, 705, 623, 691, 667, 0, 425, 397],
                            [633, 756, 463, 280, 544, 918, 1072, 425, 0, 588],
                            [898, 622, 978, 828, 356, 333, 625, 397, 588, 0]],
                           [[0, 670, 502, 866, 1429, 1114, 672, 1160, 996, 492],
                            [670, 0, 933, 922, 984, 525, 355, 962, 733, 531],
                            [502, 933, 0, 515, 1319, 1190, 719, 910, 838, 444],
                            [866, 922, 515, 0, 877, 924, 579, 412, 425, 442],
                            [1429, 984, 1319, 877, 0, 516, 765, 497, 481, 959],
                            [1114, 525, 1190, 924, 516, 0, 478, 729, 536, 751],
                            [672, 355, 719, 579, 765, 478, 0, 623, 402, 275],
                            [1160, 962, 910, 412, 497, 729, 623, 0, 230, 670],
                            [996, 733, 838, 425, 481, 536, 402, 230, 0, 506],
                            [492, 531, 444, 442, 959, 751, 275, 670, 506, 0]],
                           [[0, 922, 1084, 866, 377, 701, 583, 520, 1090, 670],
                            [922, 0, 457, 670, 1022, 908, 529, 700, 773, 310],
                            [1084, 457, 0, 1113, 1309, 729, 511, 1062, 368, 691],
                            [866, 670, 1113, 0, 690, 1290, 919, 350, 1369, 449],
                            [377, 1022, 1309, 690, 0, 1064, 862, 370, 1395, 716],
                            [701, 908, 729, 1290, 1064, 0, 395, 1037, 514, 878],
                            [583, 529, 511, 919, 862, 395, 0, 718, 534, 488],
                            [520, 700, 1062, 350, 370, 1037, 718, 0, 1233, 393],
                            [1090, 773, 368, 1369, 1395, 514, 534, 1233, 0, 921],
                            [670, 310, 691, 449, 716, 878, 488, 393, 921, 0]]
                           ]

# 路径衰减因子
path_loss_factor = 4
# 每一个ESN的信号发射功率（Watt）
esn_transmitting_power = 20
# 每一个ESN周围的高斯噪声功率（dBm，分贝毫瓦）
esn_noise_power = -100
# 边缘服务器节点的最大处理速率（tasks/s）
esn_computing_speed = []
# 每一个边缘服务器节点收到信号覆盖范围内用户服务请求的产生速率的标量因子
esn_request_coming_factor = []
# ESN-i与ESN-j间的最大信道容量（传输速率）（KB/s）
esn_transmitting_speed = []


# 边缘服务器节点
class EdgeServerNode(object):
    # 初始化
    def __init__(self, id, available_computing_speed, transmission_speed, request_coming_speed, max_delay, input_size,
                 output_size):
        # ESN id
        self.__id = id
        # ESN 收到信号覆盖范围内用户服务请求的产生速率
        self.__request_coming_speed = request_coming_speed
        # ESN 可用处理速率
        self.__available_computing_speed = available_computing_speed
        # 服务可以容忍的最大延时（ms）
        self.__max_delay = max_delay

        # ESN 与其它边缘服务器间的传输速率（发送速率）
        self.__transmission_speed = transmission_speed
        # 某一种服务的请求参数大小（KB）
        self.__input_size = input_size
        # 某一种服务的执行完毕后的返回结果数据大小（KB）
        self.__output_size = output_size

    # ESN 按当前可用服务处理速率比例分配
    def proportional_request_dispatching(self):
        # 每一个ESN的初始参与速率
        staring_computing_speed = [0.0 for i in range(ESN_NUM)]
        for j in range(ESN_NUM):
            # ESN-j 上有服务部署
            if esn_computing_speed[j] != 0.0:
                if j == self.__id:
                    # 自身节点下的传输时延+排队延时=0
                    temp1 = self.__available_computing_speed[j] - 1.0 / (self.__max_delay * math.pow(10, -3))
                    if temp1 > 0.0:
                        staring_computing_speed[j] = temp1
                else:
                    # 构造一元三次方程
                    a = self.__available_computing_speed[j]
                    b = self.__transmission_speed[self.__id][j] / self.__input_size
                    c = self.__transmission_speed[j][self.__id] / self.__output_size
                    d = self.__max_delay * math.pow(10, -3)
                    staring_computing_speed[j] = self.calculate_cubic_equation(a, b, c, d)
        # 按比例分配方案
        proportion_vector = [i / sum(staring_computing_speed) for i in staring_computing_speed]
        # 返回服务请求分发的比例向量
        return proportion_vector

    # 求解一元三次方程
    def calculate_cubic_equation(self, a, b, c, d):
        temp_A = d
        temp_B = -(a * d + b * d + c * d - 3)
        temp_C = -(2 * a + 2 * b + 2 * c - a * b * d - a * c * d - b * c * d)
        temp_D = -(a * b * c * d - a * b - a * c - b * c)
        # 利用盛金公式求解
        # 定义重根判别式
        A = temp_B ** 2 - 3 * temp_A * temp_C
        B = temp_B * temp_C - 9 * temp_A * temp_D
        C = temp_C ** 2 - 3 * temp_B * temp_D
        # 定义总判别式
        D = B ** 2 - 4 * A * C
        '''
            当A=B=0时，方程有一个三重实根。
            当Δ=B²-4AC>0时，方程有一个实根和一对共轭虚根。
            当Δ=B²-4AC=0时，方程有三个实根，其中有一个二重根。
            当Δ=B²-4AC<0时，方程有三个不相等的实根。
        '''
        temp_a, temp_b, temp_c, temp_d = a, b, c, d
        a, b, c, d = temp_A, temp_B, temp_C, temp_D
        x = []
        if A == B == 0:
            # 盛金公式1
            x.append(-1 * c / b)
        elif D > 0:
            # 盛金公式2
            Y1 = A * b + 3 * a * (1 / 2 * (-B + (D ** (1 / 2))))
            Y2 = A * b + 3 * a * (1 / 2 * (-B - (D ** (1 / 2))))
            x.append(1 / (a * 3) * (-b - Y1 ** (1 / 3) - Y2 ** (1 / 3)))
            x.append(-1)
            x.append(-1)
        elif D == 0:
            # 盛金公式3
            K = B / A
            x.append(-b / a + K)
            x.append(-K / 2)
        elif D < 0:
            # 盛金公式4
            T = (2 * A * b - 3 * a * B) / (2 * (A ** (3 / 2)))
            p = math.acos(T)
            g3 = math.cos(p / 3) - math.sin(p / 3) * (3 ** (1 / 2))
            g2 = math.cos(p / 3) + math.sin(p / 3) * (3 ** (1 / 2))
            x.append(1 / (3 * a) * (-b - 2 * (A ** (1 / 2) * math.cos(p / 3))))
            x.append(1 / (3 * a) * (-b + (A ** (1 / 2) * g2)))
            x.append(1 / (3 * a) * (-b + (A ** (1 / 2) * g3)))
        result = []
        for re in x:
            if isinstance(re, complex):
                # 保证根是实数
                continue
            else:
                if temp_a > 0 and temp_b > 0 and temp_c > 0 and re < temp_a and re < temp_b and re < temp_c and (
                        1.0 / (temp_b - re) - 1.0 / (temp_c - re)) < temp_d and re > 0:
                    result.append(re)
        if len(result) == 0:
            return 0.0
        return result[0]


# 分布式请求分发算法
def distributed_request_dispatching(esn_request_coming_speed):
    # 某一种服务可以接受的最大响应时间[200,280](ms)
    TMAX = random.randint(200, 280) + .0
    # 某一种服务的请求参数大小（KB）
    service_input_size = random.randint(300, 500) + .0
    # 某一种服务的执行完毕后的返回结果数据大小（KB）
    service_output_size = random.randint(50, 150) + .0

    # ESN-i 的平均响应时间
    esn_avg_time = [0.0 for t in range(ESN_NUM)]
    # 所有ESN服务请求分发比例向量
    esn_proportion_vector = [[0.0 for j in range(ESN_NUM)] for i in range(ESN_NUM)]
    # ESN 轮转方式计算自身的分发比例向量，多轮迭代，直至收敛条件
    # 记录迭代次数
    iterations = 1
    while True:
        # 每轮开始前，初始化累计误差
        total_sum = 0
        for i in range(ESN_NUM):
            # 对于ESN-i 系统可用的计算资源
            available_computing_speed = [0.0 for m in range(ESN_NUM)]
            for j in range(ESN_NUM):
                sum_temp = sum(
                    [esn_proportion_vector[k][j] * esn_request_coming_speed[k] for k in range(ESN_NUM) if k != i])
                available_computing_speed[j] = esn_computing_speed[j] - sum_temp
            esn_i_proportion_vector = EdgeServerNode(i, available_computing_speed, esn_transmitting_speed,
                                                     esn_request_coming_speed[i], TMAX, service_input_size,
                                                     service_output_size).proportional_request_dispatching()
            esn_proportion_vector[i] = esn_i_proportion_vector
            # 按照此时的比例分配向量，计算ESN-i 的平均响应时间
            esn_i_avg_time = sum(
                [esn_i_proportion_vector[j] * (1.0 / (
                        available_computing_speed[j] - esn_i_proportion_vector[j] * esn_request_coming_speed[i]) +
                                               (1.0 / (esn_transmitting_speed[i][j] / service_input_size -
                                                       esn_i_proportion_vector[j] * esn_request_coming_speed[
                                                           i]) if i != j else 0.0) + (1.0 / (
                                esn_transmitting_speed[j][i] / service_output_size - esn_i_proportion_vector[j] *
                                esn_request_coming_speed[i]) if i != j else 0.0)) for j in range(ESN_NUM) if
                 esn_i_proportion_vector[j] != 0.0])
            if np.any(np.array(
                    [available_computing_speed[j] - esn_i_proportion_vector[j] * esn_request_coming_speed[i] for j in
                     range(ESN_NUM)]) < 0):
                return 0, 0, 0, False
            if np.any(np.array([esn_transmitting_speed[i][j] / service_input_size - esn_i_proportion_vector[j] *
                                esn_request_coming_speed[i] if i != j else 0.0 for j in range(ESN_NUM)]) < 0):
                return 0, 0, 0, False
            if np.any(np.array([esn_transmitting_speed[j][i] / service_output_size - esn_i_proportion_vector[j] *
                                esn_request_coming_speed[i] if i != j else 0.0 for j in range(ESN_NUM)]) < 0):
                return 0, 0, 0, False

            # 计算累计误差
            total_sum += abs(esn_avg_time[i] - esn_i_avg_time)
            esn_avg_time[i] = esn_i_avg_time
        # 达到收敛条件
        if total_sum <= XI:
            break
        # 新一轮迭代
        iterations += 1
    return esn_proportion_vector, iterations, esn_avg_time, True


# 随机获取一组新的边缘服务器计算速率
def get_esn_computing_speed():
    server_types = [82, 100, 57, 60]
    esn_computing_speed = []
    for i in range(10):
        index = random.randint(0, 3)
        esn_computing_speed.append(server_types[index])
    return esn_computing_speed


# 为每一个ESN随机分配一个服务请求标量因子
def factor_creating():
    while True:
        list_temp = [random.random() for i in range(10)]
        list_temp_sum = sum(list_temp)
        # 标量因子
        factor_list = [round(i / list_temp_sum, 2) for i in list_temp]
        if sum(factor_list) == 1.0:
            if 0 not in factor_list:
                return factor_list


# 主程序入口
if __name__ == '__main__':
    finally_result_response = []
    finally_result_iter = []
    finally_result_all_esn_avg_time = []
    num_iter = 10
    for i in range(num_iter):
        esn_distance_graph = esn_distance_graph_list[i]
        esn_transmitting_speed = [[(bandwidth * math.log(
            1.0 + (esn_transmitting_power / math.pow(esn_distance_graph[i][j], 4)) / (
                    1.0 * math.pow(10, esn_noise_power / 10) * math.pow(10, -3)), 2)) / 8 / 1000
                                   if i != j else 0.0 for j in range(ESN_NUM)] for i in range(ESN_NUM)]

        esn_computing_speed = get_esn_computing_speed()
        esn_request_coming_factor = factor_creating()

        result_iter = []
        result_response = []
        result_all_esn_avg_time = []
        THETA_list = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
        theta_num = 0
        while theta_num < 17:
            THETA = THETA_list[theta_num]
            print("---------------负载率：", THETA, "--------")
            # 每一个边缘服务器节点收到信号覆盖范围内用户服务请求的产生速率（THETA的变化）
            esn_request_coming_speed = [sum(esn_computing_speed) * THETA * f for f in esn_request_coming_factor]
            iter_sum = 0
            esn_total_avg_time = 0.0
            all_esn_avg_time = [0.0 for i in range(ESN_NUM)]
            total_num = 400
            num = 0
            while num < total_num:
                # 运行分布式服务请求分发算法
                esn_proportion_vector, iterations, esn_avg_time, flag = distributed_request_dispatching(
                    esn_request_coming_speed)
                if flag is False:
                    continue
                print("-----------------ESN-i 的服务请求分发策略-------------------")
                for vector in esn_proportion_vector:
                    print(vector)
                print("本次迭代次数：", iterations)
                print("本次每一个ESN的平均响应时间：", esn_avg_time)
                all_esn_avg_time = [all_esn_avg_time[i] + esn_avg_time[i] * 1000 for i in range(ESN_NUM)]
                total_response = sum([esn_request_coming_factor[i] * esn_avg_time[i] for i in range(ESN_NUM)])
                print("本次总体平均响应时间：", total_response * 1000)
                iter_sum += iterations
                esn_total_avg_time += (total_response * 1000)
                num += 1
            all_esn_avg_time = [all_esn_avg_time[i] / total_num for i in range(ESN_NUM)]
            result_all_esn_avg_time.append(all_esn_avg_time)
            result_iter.append(iter_sum / total_num)
            result_response.append(esn_total_avg_time / total_num)
            theta_num += 1
        print("--------------- Result ---------------")
        print("不同系统负载下的迭代次数：", result_iter)
        print("不同系统负载下的总体平均响应时间：", result_response)
        finally_result_response.append(result_response)
        finally_result_iter.append(result_iter)
        finally_result_all_esn_avg_time.append(result_all_esn_avg_time)
    print("----------------最终结果------------------")
    print([k / num_iter for k in np.sum(finally_result_response, axis=0)])
    print([k / num_iter for k in np.sum(finally_result_iter, axis=0)])
    finally_result_every_esn_time = []
    for i in range(17):
        temp_list = []
        for j in range(10):
            temp_list.append(finally_result_all_esn_avg_time[j][i])
        finally_result_every_esn_time.append([k / num_iter for k in np.sum(temp_list, axis=0)])
    print(finally_result_every_esn_time)
