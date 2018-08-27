import numpy as np

def load_data(file_name):
    dataset = []
    with open(file_name) as file:
        data = file.readlines()
    for i in data:
        curline = []
        i = i.strip().split('\t')
        for j in i:
            curline.append(float(j))
        dataset.append(curline)
    return np.mat(dataset)

# 误差计算，使用欧拉距离
def distance_cal(vec_a, vec_b):
    # Numpy广播计算
    return np.sqrt(np.sum(np.power(vec_a - vec_b, 2)))

def rand_centers(dataset, k):
    n = dataset.shape[1]
    # 建立质心点，数量为k，特征数量为n
    center = np.mat(np.zeros((k, n)))
    # 随机选取每个质心点的特征值范围
    for i in range(n):
        min_i = np.min(dataset[:, i])
        range_i = float(np.max(dataset[:, i]) - min_i)
        center[:, i] = range_i * np.random.rand(k, 1) + min_i
    return center

file_name = 'Data/testSetK.txt'
dataset = load_data(file_name)
# print(rand_centers(dataset, 3))

def kMeans(dataset, k, dist_means = distance_cal, create_centers = rand_centers):
    m = dataset.shape[0]
    # 存储训练点的聚类信息，第一列存储聚类标签，第二列存储误差
    cluster_assessment = np.zeros((m,2))
    # 第一次创建质心
    centers = create_centers(dataset, k)
    cluster_changed = True
    while cluster_changed:
        cluster_changed = False
        # 遍历每个训练点，计算距离和聚类标签
        for i in range(m):
            minDistance = np.inf
            minIndex = -1
            # 依次计算训练点和质心的距离
            for j in range(k):
                distJI = dist_means(centers[j,:], dataset[i, :])
                if distJI < minDistance:
                    minDistance = distJI
                    minIndex = j
            # 只要有任意一个点没有更新聚类标签，就继续整体迭代
            if cluster_assessment[i, 0] != minIndex:
                cluster_changed = True
            # 存储训练点的聚类标签和误差平方
            cluster_assessment[i,:] = minIndex, np.power(minDistance,2)
        # print(centers)
        # 每一轮聚类完成后都调整质心位置，质心位置使用每个特征的均值
        for cent in range(k):
            ptsInclust = dataset[np.nonzero(cluster_assessment[:,0] == cent)[0], :]
            centers[cent,:] = np.mean(ptsInclust, axis=0)
    return centers, cluster_assessment


centers, cluster_assessment = kMeans(dataset, 4)


# 后处理的2种办法：（1）计算所有质心之间的距离，合并距离最近的2个点；（2）合并任意2个蔟，对比合并前后的SSE（Sum of Squared Error）变化

def b_kMeans(dataset, k, distance_means = distance_cal):
    m = dataset.shape[0]
    # 训练集聚类存储器
    cluster_assessment = np.zeros((m,2))
    # 初始单一蔟质心位置
    centers_0 = np.mean(dataset, axis=0).tolist()[0]
    # 质心列表
    centers_list = [centers_0]
    # 首次填充聚类存储器
    for j in range(m):
        # 存储训练点的距离平方
        cluster_assessment[j,1] = distance_means(centers_0, dataset[j, :])**2
    while (len(centers_list) < k):
        lowestSSE = np.inf
        # 循环每个已有蔟，进行划分尝试
        for i in range(len(centers_list)):
            ptsIncurCluster = dataset[np.nonzero(cluster_assessment[:, 0] == i)[0]]
            # 对当前蔟进行划分，存储训练点的聚类标签和误差平方
            centers, splitClustAss = kMeans(ptsIncurCluster, 2, distance_means)
            # 计算划分后的所有误差和
            sseSplit = np.sum(splitClustAss[:, 1])
            # 计算不划分的那个蔟的误差和
            sseNotSplit = np.sum(cluster_assessment[np.nonzero(cluster_assessment[:, 0] != i)[0]])
            # 计算总的误差和是否增减（也可以先计算单个蔟在划分前后的SSE之差，然后对不同蔟进行对比；还可以直接对SSE最大的蔟进行划分）
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centers
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        # 更新聚类标签，其中有一个新蔟继承原来的聚类标签
        bestClustAss[np.nonzero(bestClustAss[:, 0] == 1)[0], 0] = len(centers_list)
        bestClustAss[np.nonzero(bestClustAss[:, 0] == 0)[0], 0] = bestCentToSplit
        # 更新质心数组，其中被划分的蔟的质心被替换，并加上新蔟的质心
        centers_list[bestCentToSplit] = bestNewCents[0, :]
        centers_list.append(bestNewCents[1, :])
        # 更新聚类存储器
        cluster_assessment[np.nonzero(cluster_assessment[:, 0] == bestCentToSplit)[0], :] = bestClustAss
    return centers_list, cluster_assessment

b_centers, b_cluster_assessment = b_kMeans(dataset, 5)
print(len(b_centers))
