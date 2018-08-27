import numpy as np


def loadData():
    data_matrix = []
    label_matrix = []
    with open('Data/TestSet.txt') as file:
        for line in file.readlines():
            line_a = line.strip().split()
            data_matrix.append([1.0, float(line_a[0], float(line_a[1]))])
            label_matrix.append(int(line_a[2]))
    return data_matrix, label_matrix


def sigmoid(input_x):
    return 1.0 / (1 + np.exp(-input_x))


def gradient_boost(data_matrix, label_matrix):
    data_matrix = np.mat(data_matrix)
    label_matrix = np.mat(label_matrix).transpose()
    m, n = data_matrix.shape
    alpha = 0.001
    max_round = 500
    weights = np.ones((n, 1))
    for k in range(max_round):
        h = sigmoid(data_matrix * weights)
        error = label_matrix - h
        weights += alpha * data_matrix.transpose() * error
    return weights


def random_gradient_boost(data_matrix, label_matrix):
    m, n = data_matrix.shape
    alpha = 0.01
    weights = np.ones((n,1))
    for i in range(m):
        h = sigmoid(sum(data_matrix[i] * weights))
        error = label_matrix[i] - h
        weights += alpha * data_matrix[i] * error
    return weights


def random_gradient_boost_plus(data_matrix, label_matrix):
    m, n = len(data_matrix), len(data_matrix[0])
    alpha = 0.01
    max_round = 150
    weights = np.ones(n)
    for j in range(max_round):
        data_index = [x for x in range(m)]
        for i in range(m):
            alpha = 4 / (1 + i + j) + 0.0001
            rand_index = int(np.random.uniform(0, len(data_index)))
            h = sigmoid(sum(data_matrix[rand_index] * weights))
            error = label_matrix[rand_index] - h
            weights += alpha * error * np.array(data_matrix[rand_index])
            del data_index[rand_index]
    return weights


def classify(in_x, weights):
    prob = sigmoid(sum(in_x * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def read_data():
    file = [['Data/horseColicTraining.txt', [], []],
            ['Data/horseColicTest.txt', [], []]]
    for i in range(len(file)):
        with open(file[i][0]) as train_file:
            for line in train_file.readlines():
                line_array = []
                current_line = line.strip().split('\t')
                for j in range(21):
                    line_array.append(float(current_line[j]))
                file[i][1].append(line_array)
                file[i][2].append(float(current_line[21]))
    return file[0][1], file[0][2], file[1][1], file[1][2]


def colic_test(train_set, train_labels, test_set, test_labels):
    weights = random_gradient_boost_plus(train_set, train_labels)
    error = 0
    for i in range(len(test_labels)):
        if classify(test_set[i], weights) != test_labels[i]:
            error += 1
    return error / len(test_labels), error, len(test_labels)


train_set, train_labels, test_set, test_labels = read_data()
print(colic_test(train_set, train_labels, test_set, test_labels))

def multi_test():
    num_tests = 10
    error_rate = []
    for i in range(num_tests):
        error_rate.append(
            classify(train_set, train_labels, test_set, test_labels))
    return sum(error_rate) / len(error_rate)
