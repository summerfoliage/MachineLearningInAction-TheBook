import numpy as np
import matplotlib.pyplot as plt


def load_data(file_name):
    dataset = []
    with open(file_name) as file:
        data = file.readlines()
    for line in data:
        cur_line = []
        cur_line = line.strip().split('\t')
        dataset.append(list(map(float, cur_line)))
    return dataset


def regLeaf(dataset):
    # 以subset的平均值作为回归数
    return np.mean(dataset[:, -1])


def regErr(dataset):
    # 计算subset的方差和，以subset的离散程度作为误差分析指标
    return np.var(dataset[:, -1]) * dataset.shape[0]


def bin_split_tree(dataset, feature, value):
    # 取出2分后的subset，np.nonzero的作用是取的目标分类的行号，其中目标分类通过布尔矩阵表达
    mat0 = dataset[np.nonzero(dataset[:, feature] <= value)[0]]
    mat1 = dataset[np.nonzero(dataset[:, feature] > value)[0]]
    return np.mat(mat0), np.mat(mat1)


def create_tree(dataset, leafType=regLeaf, errTyrpe=regErr, ops=(1, 5)):
    feature, value = choose_best_split(dataset, leafType, errTyrpe, ops)
    # 到达叶节点，返回回归值
    if feature == None:
        return value
    retTree = {}
    retTree['spInd'] = feature
    retTree['spVal'] = value
    lSet, rSet = bin_split_tree(dataset, feature, value)
    retTree['left'] = create_tree(lSet, leafType, errTyrpe, ops)
    retTree['right'] = create_tree(rSet, leafType, errTyrpe, ops)
    return retTree


# test = np.mat(np.eye(4))
# print(test[:, -1].T.tolist())
# print(test[test[:, 1] > 0.5])
# print(bin_split_tree(test, 1, 0.5))
# --矩阵划分测试--
# test2 = np.mat(np.eye(4))
# mat0 = test2[:, 1] <= 0.5
# mat1 = np.nonzero(test2[:, 1] <= 0.5)[0]
# print(mat0)
# print(mat1)


def choose_best_split(dataset, leafType=regLeaf, errTyrpe=regErr, ops=(1, 5)):
    tolS = ops[0]
    tolN = ops[1]
    # 判断subset的纯度，返回叶节点
    if len(set(dataset[:, -1].T.tolist()[0])) == 1:
        print('x1')
        return None, leafType(dataset)
    m, n = dataset.shape
    S = errTyrpe(dataset)
    bestS = np.inf
    best_index = 0
    best_value = 0
    for feature_index in range(n - 1):
        for split_value in set(dataset[:, feature_index].T.tolist()[0]):
            mat0, mat1 = bin_split_tree(dataset, feature_index, split_value)
            if (mat0.shape[0] < tolN) or (mat1.shape[0] < tolN):
                # print('x2')
                # 这里不直接return的作用是，避免在循环早期就过早退出，以及当所有迭代都出现子集样本数小于tolN时，跳过这个代码块，真正执行的会是x4
                continue
            newS = errTyrpe(mat0) + errTyrpe(mat1)
            if newS < bestS:
                best_index = feature_index
                best_value = split_value
                bestS = newS
    # 判断误差减少量，决定是否采用上一步骤的划分
    # 如果效果未达到，返回叶节点
    if (S - bestS) < tolS:
        # print('x3')
        return None, leafType(dataset)
    # 使用嵌套循环中找到的best_feature，best_value进行实际划分
    mat0, mat1 = bin_split_tree(dataset, best_index, best_value)
    # x1，x2，x3都是前剪枝的案例，需要用户手动指定参数集合ops
    # 这里的作用是当所有循环都出现划分后的子集样本数小于tolN时，x2的代码块不会执行，此时按照默认的best_index和best_value进行数据的划分
    if (mat0.shape[0] < tolN) or (mat1.shape[0] < tolN):
        # print('x4')
        return None, leafType(dataset)
    return best_index, best_value


# file_name = 'Data/ex0.txt'
# dataset = np.mat(load_data(file_name))
# regression_tree = create_tree(dataset)
# print(regression_tree)

# plt.scatter(dataset[:,0].tolist(), dataset[:,1].tolist())
# plt.show()

# with open('regression_tree.txt', 'w') as file:
#     file.write(str(regression_tree))


def isTree(obj):
    return (type(obj).__name__ == 'dict')


def getMean(tree):
    # 找到树最右边的叶节点
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    # 找到树最左边的叶节点
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    # return tree['left'], tree['right']
    return (tree['left'] + tree['right']) / 2.0

# print(getMean(regression_tree))


def prune(tree, testData):
    # if testData.shape[0] == 0: return getMean(tree)
    # 使用字典树对测试数据进行划分
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet = bin_split_tree(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    # 对叶节点进行计算
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = bin_split_tree(testData, tree['spInd'], tree['spVal'])
        # 需要和字典树的标准误方差做比较
        # 计算没有合并叶节点时的方差求和和字典树标准方差求和的差异
        errorNoMerge = np.sum(np.power(
            lSet[:, -1] - tree['left'], 2)) + np.sum(np.power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right']) / 2.0
        # 计算合并后，叶节点的方差和和枝节点方差和的差异
        errorMerge = np.sum(np.power(testData[:, -1] - treeMean, 2))
        if errorMerge < errorNoMerge:
            print('merging')
            return treeMean
        else:
            return tree
    else:
        return tree

# 虽然create_tree函数里面已经有通过比较dataset和subset的误差平方和判断是否分裂的代码块，这里的后剪枝过程使用另外的测试集进行同样的比较，进而进行合并处理
# test_dataset = np.mat(load_data('Data/ex2test.txt'))
# print(prune(regression_tree, test_dataset))

# 线性回归的正规方差解


def linear_solve(dataset):
    m, n = dataset.shape
    x = np.mat(np.ones((m, n)))
    y = np.mat(np.ones((m, 1)))
    # 下面x的index从1:n，表示的是x是个2列数据集，其中第一列为1，对应etha0；第二列为x1，对应ehta1
    x[:, 1:n] = dataset[:, 0:n - 1]
    y = dataset[:, -1]
    xTx = x.T * x
    ws = xTx.I * (x.T * y)
    return ws, x, y

# print(linear_solve(dataset))

# 返回回归系数作为叶节点的值


def modelLeaf(dataset):
    ws, x, y = linear_solve(dataset)
    return ws


def modelErr(dataset):
    ws, x, y = linear_solve(dataset)
    y_hat = x * ws
    return np.sum(np.power(y - y_hat, 2))


file_name2 = 'Data/exp2.txt'
dataset2 = np.mat(load_data(file_name2))
model_tree = create_tree(dataset2, modelLeaf, modelErr, (1, 10))
print(model_tree)


# 这里的model或者为回归树的叶节点（平均值），或者为模型树的叶节点（线性回归的ehta参数），取值为tree['left']或tree['right']
def regression_tree_evaluate(model, input_x):
    return float(model)


def model_tree_evaluate(model, input_x):
    n = input_x.shape[1]
    X = np.mat(np.ones((1, n + 1)))
    X[:, 1:n + 1] = input_x
    # 由于模型树返回的是etha，所以用X矩阵乘以ehta得到y
    return float(X * model)


def tree_predict(tree, input_x, modelEval=regression_tree_evaluate):
    # 下面的if只适用于根节点就是叶节点的情况
    if not isTree(tree):
        return modelErr(tree, input_x)
    if input_x[tree['spInd']] <= tree['spVal']:
        if isTree(tree['left']):
            return tree_predict(tree['left'], input_x, modelEval)
        else:
            return modelEval(tree['left'], input_x)
    else:
        if isTree(tree['right']):
            return tree_predict(tree['right'], input_x, modelEval)
        else:
            return modelEval(tree['right'], input_x)


def create_predict(tree, test_dataset, modelEval=regression_tree):
    m = len(test_dataset)
    y_hat = np.mat(np.zeros((m, 1)))
    for i in range(m):
        y_hat[i, 0] = tree_predict(tree, np.mat(test_dataset[i]), modelEval)
    return y_hat
