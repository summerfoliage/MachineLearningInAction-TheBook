import numpy as np

# datasets = [[1., 2.1], [2., 1.1], [1.3, 1.], [1., 1.], [2., 1.]]

# labels = [1.0, 1.0, -1.0, -1.0, 1.0]


def load_datasets(filename):
    datasets = []
    labels = []
    length = 0
    with open(filename) as file:
        length = len(file.readline().strip().split('\t'))
        data = file.readlines()
    for i in data:
        sub_data = []
        i = i.strip().split('\t')
        for j in range(length - 1):
            sub_data.append(float(i[j]))
        labels.append(float(i[length - 1]))
        datasets.append(sub_data)
    return datasets, labels


datasets, labels = load_datasets('Data/horseColicTraining2.txt')
D = np.mat(np.ones((21, 1)) / 21)


def tree_classify(datasets, best_feature, threshold_value, threshold_direction):
    return_labels = np.ones((datasets.shape[0], 1))
    if threshold_direction == 'left':
        return_labels[datasets[:, best_feature] <= threshold_value] = -1.0
    else:
        return_labels[datasets[:, best_feature] > threshold_value] = -1.0
    return return_labels


def build_best_weak_tree(datasets, labels, D):
    datasets_mat = np.mat(datasets)
    labels_mat = np.mat(labels).T
    m, n = datasets_mat.shape
    step = 10  # 搜索步长
    best_weak_tree = {}
    best_weak_tree_result = np.mat(np.zeros((m, 1)))
    error_rate = 1
    for i in range(n):
        range_min = datasets_mat[:, i].min()
        range_max = datasets_mat[:, i].max()
        step_size = (range_max - range_min) / step
        for j in range(0, step + 1):
            for threshold_direction in ['left', 'right']:
                threshold_value = range_min + j * step_size
                predict_labels = tree_classify(
                    datasets_mat, i, threshold_value, threshold_direction)
                error = np.mat(np.zeros((m, 1)))
                error[predict_labels != labels_mat] = 1
                weight_error = D.T * error
                if weight_error < error_rate:
                    error_rate = weight_error
                    best_weak_tree_result = predict_labels
                    best_weak_tree['best_feature'] = i
                    best_weak_tree['threshold_value'] = threshold_value
                    best_weak_tree['threshold_direction'] = threshold_direction
    return best_weak_tree, error_rate, best_weak_tree_result

#print(datasets, labels, D)
#print(build_best_weak_tree(datasets, labels, D))


def adaboost_training(datasets, labels, turns=5000):
    best_weak_tree_collections = []
    m = np.mat(datasets).shape[0]
    D = np.mat(np.ones((m, 1)) / m)
    agg_predict_labels = np.mat(np.zeros((m, 1)))
    for i in range(turns):
        best_weak_tree, error_rate, best_weak_tree_result = build_best_weak_tree(
            datasets, labels, D)
        alpha = float(0.5 * np.log((1 - error_rate) / error_rate))
        best_weak_tree['alpha'] = alpha
        best_weak_tree_collections.append(best_weak_tree)
        # 更新权重D
        expon = np.multiply(-1 * alpha * np.mat(labels).T,
                            best_weak_tree_result)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()
        # 计算之前每一次预测结果的权重，为后续加权求和做准备
        agg_predict_labels += alpha * best_weak_tree_result
        agg_errors = np.multiply(
            np.sign(agg_predict_labels) != np.mat(labels).T, np.ones((m, 1)))
        error_rate = agg_errors.sum() / m
        if error_rate == 0:
            break
    return best_weak_tree_collections


def adaboost_classify(data, adaboost_classifier):
    predict_labels = np.mat(np.zeros((data.shape[0], 1)))
    for i in range(len(adaboost_classifier)):
        predict_labels += tree_classify(data, adaboost_classifier[i]['best_feature'], adaboost_classifier[i]
                                        ['threshold_value'], adaboost_classifier[i]['threshold_direction']) * adaboost_classifier[i]['alpha']
    return np.sign(predict_labels)


adaboost_classifier = adaboost_training(datasets, labels)
error_count = np.zeros((len(datasets), 1))
error_count[adaboost_classify(
    np.mat(datasets), adaboost_classifier) != np.mat(labels).T] = 1
print(error_count.sum() / len(datasets), error_count.sum())
