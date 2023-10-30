import random
import csv
import numpy as np


num_info = 100
info_list = []

for i in range(num_info):
    f_max = 1.5 + np.random.rand()
    f_min = 0.005
    B_max = 5
    B_min = 1
    f_num_samples = 600
    f_step = 0.002
    # 计算可选值的数量
    num_values = int((f_max - f_min) / f_step)
    # 生成可选值列表
    values = [f_min + i * f_step for i in range(num_values)]
    # 使用random.sample()函数从可选值列表中随机选择指定数量的值
    fList = random.sample(values, f_num_samples)
    fList.sort()

    info_list.append({
        "C": np.power(10, -27.0) * np.random.rand() * 0.5,
        "N": np.power(10, -9.0) * np.random.rand() * 0.5,
        "fList": fList,
    })

header = ['C', 'N', 'fList']

with open('data.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()
    writer.writerows(info_list)