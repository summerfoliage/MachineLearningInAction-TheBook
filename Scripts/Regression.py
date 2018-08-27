import numpy as np


def load_data(file_name):
    feature_num = 0
    datasets = []
    labels = []
    with open(file_name) as file:
        data = file.readlines()
    feature_num = len(data[0].strip().split('\t'))
    for i in data:
        cur_line = []
        i = i.strip().split('\t')
        for j in range(feature_num - 1):
            cur_line.append(float(i[j]))
        labels.append(float(i[feature_num - 1]))
        datasets.append(cur_line)
    return datasets, labels


def stand_regression(x_array, y_array):
    x_mat = np.mat(x_array)
    y_mat = np.mat(y_array)
    xTx = x_mat.T * x_mat
    # if np.linalg.det(xTx) == 0:
    #     print('Sorry')
    #     return
    ws = np.linalg.solve(xTx, x_mat.T * y_mat.T)
    print(ws)
    return ws

file_name = 'Data/ex0.txt'
x_array, y_array = load_data(file_name)
x_mat = np.mat(x_array)
y_mat = np.mat(y_array)
ws = stand_regression(x_array, y_array)

# 通过计算相关系数的方法检验回归的匹配程度
y_new = x_mat * ws
print(np.corrcoef(y_new.T, y_mat))
