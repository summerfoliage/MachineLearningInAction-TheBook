import numpy as np


def loadDataset(fileName, delim='\t'):
    with open(fileName) as file:
        stringLine = [line.strip().split(delim) for line in file .readlines()]
        dataset = [map(float, line) for line in stringLine]
        return np.mat(dataset)


def pca(dataset, topNfeat=np.inf):
    # 矩阵去均值
    meanVals = np.mean(dataset, axis=0)
    meanRemoved = dataset - meanVals
    # 求出协方差矩阵
    covMat = np.cov(meanRemoved, rowvar=0)
    # 求出协方差矩阵的特征值和特征向量
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    # 特征值排序
    # np.argsort是按元素从小到大排列index
    eigValInd = np.argsort(eigVals)
    eigValInd = eigValInd[:-(topNfeat + 1):-1]
    redEigVects = eigVects[:, eigValInd]
    # 计算矩阵相乘得到降维矩阵
    lowDDatMat = meanRemoved * redEigVects
    reconMat = (lowDDatMat * redEigVects.T) + meanVals
    return lowDDatMat, reconMat
