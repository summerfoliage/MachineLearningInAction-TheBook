class tree_node:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}

    def inc(self, numOccur):
        self.count += numOccur

    def disp(self, ind=0):
        print('    ' * ind, self.name, '    ', self.count)
        for child in self.children.values():
            child.disp(ind + 1)


# rootNode = tree_node('pyramid', 9, None)
# rootNode.children['eye'] = tree_node('eye', 13, None)
# rootNode.children['phoenix'] = tree_node('phenix', 3, None)
# rootNode.disp()
# print(rootNode.children)


def create_tree(dataset, minsup=1):
    headerTable_0 = {}
    headerTable = {}
    # 生成headerTable
    for trans in dataset:
        for item in trans:
            headerTable_0[item] = headerTable_0.get(item, 0) + dataset[trans]
    headerTable = headerTable_0.copy()
    for k in headerTable_0.keys():
        if headerTable_0[k] < minsup:
            del(headerTable[k])
    # 生成单元素集
    freqItemSet = set(headerTable.keys())
    if len(freqItemSet) == 0:
        return None, None
    # print('headerTable: ', headerTable)
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]
    # print('headerTable: ', headerTable)
    # 生成fp树根节点
    retTree = tree_node('Null Set', 1, None)
    for tranSet, count in dataset.items():
        localD = {}
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        # print('localD: ', localD)
        # 对单元素项集进行排序
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(
                localD.items(), key=lambda p:p[1], reverse=True)]
            # print('orderedItems: ', orderedItems)
            updateTree(orderedItems, retTree, headerTable, count)
            # retTree.disp()
    return retTree, headerTable


def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count)
    else:
        inTree.children[items[0]] = tree_node(items[0], count, inTree)
        if headerTable[items[0]][1] == None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)


def updateHeader(nodeToTest, targetNode):
    while (nodeToTest.nodeLink != None):
        # 形参的改变不影响实参
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


def loadData():
    dataset = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return dataset


def createInitSet(dataset):
    retDict = {}
    for trans in dataset:
        retDict[frozenset(trans)] = 1
    return retDict


dataset = loadData()
initSet = createInitSet(dataset)
fpTree, headerTable = create_tree(initSet, 3)
fpTree.disp()
# print(headerTable)

# 发现条件模式基，即以目标单项集为叶节点的子树
def ascendTree(leafNode, prefixPath):
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)


def findPrefixPath(basePat, treeNode):
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        # print(prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats

# print(findPrefixPath('r', headerTable['r'][1]))


def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p:p[0])]
    for basePat in bigL:
        print(bigL)
        print(basePat)
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        conPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        myCondTree, myHead = create_tree(conPattBases, minSup)
        if myHead != None:
            print('conditional tree for: ', newFreqSet)
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)
    return freqItemList


freqItems = []
freqItems = mineTree(fpTree, headerTable, 3, set([]), freqItems)
print(freqItems)
