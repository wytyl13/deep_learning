# -*- coding: utf-8 -*-
#*****************************************************************
#   Copyright (C) 2023 IEucd Inc. All rights reserved.
#   
#   @Author: weiyutao
#   @Created Time : 2023/2/28 16:46:43
#   @File Name : test.py
#   @Description : 
#
#*****************************************************************
import time


# print(int(time.mktime(time.localtime(time.time()))))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    # 假设有8个类别和10个年份的数据
    categories = ['Category 1', 'Category 2', 'Category 3', 'Category 4', 'Category 5',
                'Category 6', 'Category 7', 'Category 8']
    years = np.arange(2013, 2023)  # 假设从2013年到2022年

    # 假设数据是随机生成的，您可以替换这里的数据为您自己的数据
    data = np.random.randint(10, 50, size=(8, 10))  # 8行（类别），10列（年份）

    # 设置图形大小
    plt.figure(figsize=(12, 6))

    # 设置并列柱状图的宽度
    bar_width = 0.15

    # 设置位置偏移，使得八个类别的柱状图并列显示
    bar_positions = np.arange(len(years))
    for i in range(len(categories)):
        plt.bar(bar_positions + i * bar_width, data[i], width=bar_width, label=categories[i])

    # 设置X轴刻度和标签
    plt.xticks(bar_positions + (len(categories) - 1) * bar_width / 2, years)

    # 添加标题和图例
    plt.title('Sample Grouped Bar Chart (10 Years)', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.legend()

    # 显示图表
    plt.tight_layout()
    plt.show()