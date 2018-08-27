import math
from collections import Counter, defaultdict
import operator
import pickle


# 计算熵
def cal_entropy(dataset):
    l_dataset = len(dataset)
    entropy = 0
    label_count = defaultdict(int)
    for i in dataset:
        label_count[i[-1]] += 1
    for k, v in label_count.items():
        p = v / l_dataset
        entropy -= p * math.log(p, 2)
    return entropy

# 创造数据集


def create_dataset():
    dataset = [[1, 1, 1, 'maybe'], [1, 1, 0, 'yes'], [1, 0, 1, 'no'], [0, 1, 0, 'no'], [
        0, 1, 1, 'no'], [0, 0, 1, 'yes'], [0, 0, 0, 'no'], [1, 0, 0, 'no']]
    labels = ['no surfacing', 'flippers', 'swiming']
    return dataset, labels

# dataset, labels = create_dataset()
# print(cal_entropy(dataset))

# 划分数据集


def split_dataset(dataset, feature_index, value):
    new_dataset = []
    for item in dataset:
        if item[feature_index] == value:
            sub_item = item[:feature_index]
            sub_item.extend(item[feature_index + 1:])
            new_dataset.append(sub_item)
    return new_dataset

# dataset, labels = create_dataset()
# print(dataset, '\n')
# print(split_dataset(dataset, 1, 1))

# 通过遍历每个特征的每个值，计算不同特征的熵的求和


def best_feature_to_split(dataset):
    l_feature = len(dataset[0]) - 1
    base_entropy = cal_entropy(dataset)
    best_feature = 0
    best_info_gain = 0
    for i in range(l_feature):
        feature_list = [iterm[i] for iterm in dataset]
        unique_value = set(feature_list)
        # print(i, unique_value)
        new_entropy = 0.0
        for value in unique_value:
            sub_dataset = split_dataset(dataset, i, value)
            # 熵越小越好，0代表信息是最纯的
            p = len(sub_dataset) / float(len(dataset))
            new_entropy += p * cal_entropy(sub_dataset)
            # print(i, value, new_entropy, cal_entropy(sub_dataset))
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature

# dataset, labels = create_dataset()
# print(dataset, '\n')
# print(best_feature_to_split(dataset))

# 在树分裂结束时，对熵不为0的叶节点进行标签投票


def major_count(class_list):
    class_count = defaultdict(int)
    for iterm in class_list:
        class_count += 1
    sorted_class_count = sorted(
        class_count.iteritems(), key=lambda s: s[1], reverse=True)
    return sorted_class_count[0][0]

# 只适用于离散型特征，每个特征只使用一次


def create_tree(dataset, labels):
    class_list = [iterm[-1] for iterm in dataset]
    # 递归结束条件1：叶节点内所有数据为统一类别
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # 递归结束条件2：每遍历一次，裁剪一次特征
    if len(dataset[0]) == 1:
        return major_count(class_list)
    # 树分裂计算
    best_feature = best_feature_to_split(dataset)
    best_feature_label = labels[best_feature]
    # 每次递归时，my_tree在函数实例中都是上一层my_tree的值
    my_tree = {best_feature_label: {}}
    del(labels[best_feature])
    # 得到最佳特征的所有去重特征值，字典用for取key
    feature_values = set([iterm[best_feature] for iterm in dataset])
    # 递归计算sub_dataset和sub_labels
    for value in feature_values:
        # 在遍历第一个value时，如果不复制，那么labels就会被改变，从而影响第二个value的计算
        sub_labels = labels[:]
        my_tree[best_feature_label][value] = create_tree(
            split_dataset(dataset, best_feature, value), sub_labels)
    return my_tree

# dataset, labels = create_dataset()
# my_tree = create_tree(dataset, labels)
# print(my_tree)


def classify(input_tree, feature_labels, dataset):
    # 取出字典树的第一个key
    first = list(input_tree.keys())[0]
    # 取出字典树的第一个value
    second = input_tree[first]
    # 根据字典树的第一个key，取出对应的特征列表index
    feature_index = feature_labels.index(first)
    # 由于树字典树的第一个value是下一层的字典，所以遍历下一层字典的key，找到进入的分支
    for key in second.keys():
        # 从dataset中找到对应的数据列，如果特征值和树节点相同则继续计算，不相同则没必要计算
        if dataset[feature_index] == key:
            # 判断下一层的value是否是个字典，如果是则进行递归
            if isinstance(second[key], dict):
                class_label = classify(second[key], feature_labels, dataset)
            else:
                # 如果是树的叶节点，则返回节点的value
                class_label = second[key]
    return class_label


dataset, labels = create_dataset()
labels2 = labels[:]
my_tree = create_tree(dataset, labels)
print(my_tree)
print(classify(my_tree, labels2, [1, 1, 1]))


def store_tree(my_tree, filename):
    with open(filename, 'wb') as file:
        pickle.dump(my_tree, file)


def retrive_tree(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# store_tree(my_tree, 'a_tree.txt')
# print(retrive_tree('a_tree.txt'))

# lenses_labels = ['age', 'prescript', 'astigmatic', 'tearrate']
# # 快速读取文件，存储为列表
# with open('Data/lenses.txt') as file:
#     lenses = [line.strip().split('\t') for line in file.readlines()]
# lenses_trees = create_tree(lenses, lenses_labels)
# print(lenses_trees)

# ID3算法无法直接处理数值型数据，但可以通过量化的方法将数值型数据转化为离散型数据
# 书里第2、3章讨论的是非参数方法，即预测数据会被明确划分到某个分类之中
# ID3算法的流程是：1. 测量数据集的熵；2. 寻找最优办法划分数据集；3. 递归处理数据集，直到不可划分或者熵最小
