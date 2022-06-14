# -*- coding: utf-8 -*-
import math
import random
import numpy as np

'''
DRDG方法，在不同系统负载率下，总体平均响应时间的变化
'''

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

    # ESN 从自身利益最大化即最小化平均响应时间出发，进行请求分发决策
    def optimal_request_dispatching(self):
        # 服务请求分发比例向量初始化，即ESN-i 对 ESN-j的分发比例向量
        proportion_vector = [0.0 for i in range(ESN_NUM)]
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
        # 对所有的ESN-j的初始参与速率降序排序
        # 保存排序前的索引，采用字典形式
        keys = list(range(len(staring_computing_speed)))
        dict_temp = dict(zip(keys, staring_computing_speed))
        # 降序后的速率，是一个二元组（index,speed）组成的元组列表，index代表速率speed排序前的索引
        staring_computing_speed_desc = sorted(dict_temp.items(), key=lambda x: x[1], reverse=True)
        # 开始计算每一个ESN-i的请求分发比例向量
        speed_temp = (sum([item[1] for item in staring_computing_speed_desc]) - self.__request_coming_speed) / len(
            staring_computing_speed_desc)
        while speed_temp > staring_computing_speed_desc[-1][1]:
            staring_computing_speed_desc.pop()
            speed_temp = (sum([item[1] for item in staring_computing_speed_desc]) - self.__request_coming_speed) / len(
                staring_computing_speed_desc)
        if speed_temp < 0:
            return 0, False
        for item in staring_computing_speed_desc:
            proportion_vector[item[0]] = (item[1] - speed_temp) / self.__request_coming_speed
        # 返回服务请求分发的比例向量
        return proportion_vector, True

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
        # 每个ESN从自身利益最大化角度进行请求分发决策
        for i in range(ESN_NUM):
            # 对于ESN-i 系统可用的计算资源
            available_computing_speed = [0.0 for m in range(ESN_NUM)]
            for j in range(ESN_NUM):
                sum_temp = sum(
                    [esn_proportion_vector[k][j] * esn_request_coming_speed[k] for k in range(ESN_NUM) if k != i])
                available_computing_speed[j] = esn_computing_speed[j] - sum_temp
            esn_i_proportion_vector, flag = EdgeServerNode(i, available_computing_speed, esn_transmitting_speed,
                                                           esn_request_coming_speed[i], TMAX, service_input_size,
                                                           service_output_size).optimal_request_dispatching()
            if flag is False:
                return 0, 0, 0, False
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

