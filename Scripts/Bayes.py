import numpy as np
import math


def load_data():
    mail_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'], ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'], ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'], ['stop', 'posting', 'stupid', 'worthless', 'garbage'], ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'], ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_list = [0, 1, 0, 1, 0, 1]
    return mail_list, class_list


def build_word_pack(dataset):
    word_pack = set()
    for mail in dataset:
        word_pack = word_pack | set(mail)
    return sorted(word_pack)


def create_word_vector(word_pack, input_mail):
    word_vector = np.zeros(len(word_pack))
    for word in input_mail:
        if word in word_pack:
            word_vector[word_pack.index(word)] += 1
    return word_vector


mail_list, class_list = load_data()
word_pack = build_word_pack(mail_list)
word_matrix = []
for i in range(len(mail_list)):
    word_matrix.append(create_word_vector(word_pack, mail_list[i]))


def trainNB(word_matrix, class_list):
    pc1 = sum(class_list) / len(class_list)
    pc0 = 1 - pc1
    # 使所有word在不同class的初始P(W|C)都是50％
    w1 = np.ones(len(word_matrix[0]))
    w0 = np.ones(len(word_matrix[0]))
    c1 = 2
    c0 = 2
    for i in range(len(word_matrix)):
        if class_list[i] == 1:
            w1 += word_matrix[i]
            c1 += len(word_matrix[i])
        else:
            w0 += word_matrix[i]
            c0 += len(word_matrix[i])
    pwc1 = np.log(w1 / c1)
    pwc0 = np.log(w0 / c0)
    return pwc1, pwc0, pc1


pwc1, pwc0, pc1 = trainNB(word_matrix, class_list)


def classifyNB(test_mail, pwc1, pwc0, pc1, word_pack):
    test_vector = create_word_vector(word_pack, test_mail)
    pcw0 = np.sum(test_vector * pwc0) + math.log(1 - pc1)
    pcw1 = np.sum(test_vector * pwc1) + math.log(pc1)
    if pcw0 > pcw1:
        return 0
    else:
        return 1

# a_test = ['love', 'my', 'dalmation']
# b_test = ['stupid', 'garbage']
# print(classifyNB(a_test, pwc1, pwc0, pc1))
# print(classifyNB(b_test, pwc1, pwc0, pc1))


def textParse(long_mail):
    import re
    listOfTokens = re.split(r'\w*', long_mail)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def classify_spam_mail():
    import random
    class_list = []
    mail_list = []
    for i in range(1, 26):
        word_list = textParse(
            open('Datasets/email/spam/{}.txt'.format(i)).read())
        class_list.append(1)
        mail_list.append(word_list)
        word_list = textParse(
            open('Datasets/email/ham/{}.txt'.format(i)).read())
        class_list.append(0)
        mail_list.append(word_list)
    word_pack = build_word_pack(mail_list)
    train_set = list(range(50))
    test_set = []
    for i in range(10):
        index = int(random.uniform(1, len(train_set)))
        test_set.append(index)
        del train_set[index]
    train_matrix = []
    train_class = []
    for i in train_set:
        train_matrix.append(create_word_vector(word_pack, mail_list[i]))
        train_class.append(class_list[i])
    p1, p0, ps = trainNB(train_matrix, train_class)
    error = 0
    for j in test_set:
        if classifyNB(mail_list[j], p1, p0, ps, word_pack) != class_list[j]:
            error += 1
    return error / len(test_set)

# average_error = 0
# for i in range(10):
#     average_error += classify_spam_mail()
# print(average_error/10)
