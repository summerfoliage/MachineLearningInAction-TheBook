from collections import defaultdict, Counter


def load_dataset():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5], [1, 4, 6], [4, 2], [2, 6], [6, 5]]

# 首次创建atom和set


def createC(dataset):
    atom = []
    for transaction in dataset:
        for item in transaction:
            if [item] not in atom:
                atom.append([item])
    atom.sort()
    return list(map(frozenset, atom)), list(map(set, dataset))


dataset = load_dataset()
atoms, sets = createC(dataset)
# print(list(atoms), list(sets))

# 通过计算atom在set中的支持度，并进行过滤低频项目


def scanD(sets, atoms, minSupport):
    numOfAtoms = defaultdict(int)
    for item in sets:
        for atom in atoms:
            if atom.issubset(item):
                numOfAtoms[atom] += 1
    numOfItems = len(sets)
    targetAtoms = []
    supportData = {}
    for atom in numOfAtoms:
        support = numOfAtoms[atom] / numOfItems
        if support >= minSupport:
            targetAtoms.append(atom)
            supportData[atom] = support
    return targetAtoms, supportData


# targetAtoms, supportData = scanD(sets, atoms, 0.5)
# print(targetAtoms, supportData)

# 从k-1生成长度为k的项目


def aprioriGen(targetAtoms, k):
    targetSets = []
    lenOfAtoms = len(targetAtoms)
    for i in range(lenOfAtoms):
        for j in range(i + 1, lenOfAtoms):
            l1 = list(targetAtoms[i])[:k - 2]
            # print(l1)
            l2 = list(targetAtoms[j])[:k - 2]
            # print(l2)
            l1.sort()
            l2.sort()
            if l1 == l2:
                targetSets.append(targetAtoms[i] | targetAtoms[j])
            # targetSets.append(targetAtoms[i] | targetAtoms[j])
    # return set(targetSets)
    return targetSets

# print(targetAtoms)
# a_targetAtoms= aprioriGen(targetAtoms, 2)
# print(a_targetAtoms)
# b_targetSets = aprioriGen(a_targetAtoms, 3)
# print(b_targetSets)
# c_targetSets = aprioriGen(b_targetSets, 4)
# print(c_targetSets)


def apriori(dataset, minSupport=0.2):
    atoms, sets = createC(dataset)
    targetAtoms, supportData = scanD(sets, atoms, minSupport)
    L = [targetAtoms]
    k = 2
    # 迭代组合过程，k代表一个项集里包含的元素数目，最大的k为所有单个元素的数目K；迭代的停止条件是没有新的包含k个元素的频繁项集被生成
    while (len(L[k - 2]) > 0):
        new_targetSets = aprioriGen(L[k - 2], k)
        new_targetAtoms, new_supportData = scanD(
            sets, new_targetSets, minSupport)
        supportData.update(new_supportData)
        L.append(new_targetAtoms)
        k += 1
    return L, supportData


L, supportData = apriori(dataset)
print(L, '\n', supportData)
result = str([L, '\n', supportData])
with open('apriori.txt', 'w') as file:
    file.write(result)

def generateRule(L, supportData, minConf=0.5):
    bigRuleList = []
    for i in range(1, len(L)):
        for analyseSet in L[i]:
            sub_freqSet = [frozenset([item]) for item in analyseSet]
            if (i > 1):
                calcConf(analyseSet, sub_freqSet,
                         supportData, bigRuleList, minConf)
                rulesFromConsqe(analyseSet, sub_freqSet,
                                supportData, bigRuleList, minConf)
            else:
                calcConf(analyseSet, sub_freqSet,
                         supportData, bigRuleList, minConf)
                pass
    return bigRuleList


def calcConf(analyseSet, sub_freqSet, supportData, br1, minConf=0.7):
    # 下面的prunedH有必要，因为需要通过判断是否满足最小可信度来过滤频繁项集
    prunedH = []
    for affectSet in sub_freqSet:
        # 基于某个项集是频繁的，那么它的所有子集也是频繁的，子集的子集也是频繁的，所以这个位置不需要进行支持度过滤
        conf = supportData[analyseSet] / supportData[analyseSet - affectSet]
        if conf >= minConf:
            print(analyseSet - affectSet, '--->', affectSet, 'connf:', conf)
            br1.append((analyseSet - affectSet, affectSet, conf))
            prunedH.append(affectSet)
    # print('A',sub_freqSet)
    # print('B',prunedH)
    return prunedH


def rulesFromConsqe(analyseSet, sub_freqSet, supportData, br1, minConf=0.7):
    m = len(sub_freqSet[0])
    # 首次执行时，sub_freqset[0]必定是长度只有1的，这时可以直接计算conf
    # 下面的if用于判断analyseSet的总长度和sub_freqSet单个元素的长度对比，即超过2位时执行
    if (len(analyseSet) > (m + 1)):
        Hmp1 = aprioriGen(sub_freqSet, m + 1)
        # 基于某个项集是频繁的，那么它的所有子集也是频繁的，子集的子集也是频繁的，所以这个位置不需要进行支持度过滤
        Hmp1 = calcConf(analyseSet, Hmp1, supportData, br1, minConf)
        # 对sub_freqSet进行组合升级，并返回升级后的sub_freqSet
        # 下面的if用于判断组合后的sub_freqSet非空，否则就不需要执行了
        if(len(Hmp1) > 1):
            rulesFromConsqe(analyseSet, Hmp1, supportData, br1, minConf)


rule = generateRule(L, supportData)
print('**Rule**', '\n', rule)
