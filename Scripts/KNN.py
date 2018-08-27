import numpy as np
import operator


def classify(in_x, dataset, labels, k):
    # 取得训练集的行数
    dataset_size = dataset.shape[0]
    # 计算欧拉距离（应用到了np的广播）
    distances = (
        ((np.tile(in_x, (dataset_size, 1)) - dataset)**2).sum(axis=1))**0.5
    # 取得距离排名
    sorted_distances = np.argsort(distances)
    class_count = {}
    for i in range(k):
        # 取得距离排名前k的labels
        vote_labels = labels[sorted_distances[i]]
        # print(vote_labels)
        # labels计数
        class_count[vote_labels] = class_count.get(vote_labels, 0) + 1
    # labels排名（注意sorted的用法，以及key=lambda s:s[1]）
    sorted_class_count = sorted(
        class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def file2matrix(url):
    with open(url) as file:
        data = file.readlines()
    X_train = np.zeros((len(data), 3))
    y_train = []
    index = 0
    for i in data:
        i = i.strip()
        i = i.split('\t')
        X_train[index, :] = i[0:3]
        y_train.append(i[-1])
        index += 1
    return X_train, y_train


def auto_norm(dataset):
    min_x = dataset.min(0)
    max_x = dataset.max(0)
    range_x = max_x - min_x
    l = dataset.shape[0]
    new_data = (dataset - np.tile(min_x, (l, 1))) / np.tile(range_x, (l, 1))
    return new_data


def dating_test(url):
    X_train, y_train = file2matrix(url)
    X_train = auto_norm(X_train)
    m = X_train.shape[0]
    n = int(m * 0.1)
    error = 0
    for i in range(n):
        test_result = classify(X_train[i, :], X_train[n:m, :], y_train[n:m], 3)
        if test_result != y_train[i]:
            error += 1
            print(test_result, y_train[i])
    return error / m, error


def img2vector(filename):
    with open(filename) as file:
        my_vector = np.zeros((1, 1024))
        for i in range(32):
            data = file.readline().strip()
            for j in range(32):
                my_vector[0, 32 * i + j] = int(data[j])
    return my_vector


from os import listdir


def hand_writing_classifier(train_path, test_path, k):
    hw_y = []
    test_y = []
    error = 0
    hw_result = []
    filelist = listdir(train_path)
    testlist = listdir(test_path)
    file_l = len(filelist)
    test_l = len(testlist)
    hw_X = np.zeros((file_l, 1024))
    test_X = np.zeros((test_l, 1024))
    for i, j in enumerate(filelist):
        hw_y.append(int(j[0]))
        hw_X[i] = img2vector('Data/digits/trainingDigits/' + j)
    for i, j in enumerate(testlist):
        test_y.append(int(j[0]))
        test_X[i] = img2vector('Data/digits/testDigits/' + j)
        hw_result.append(classify(test_X[i], hw_X, hw_y, k))
        if hw_result[i] != test_y[i]:
            error += 1
    return k, error, error / test_l


file_path = 'Data/digits/trainingDigits'
test_path = 'Data/digits/testDigits'
for k in range(3, 11):
    print(hand_writing_classifier(file_path, test_path, k))
