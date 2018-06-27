# -*- coding: utf-8 -*-
import jieba
from scipy import spatial
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def calSim(word1, word2):

    if w2v.has_key(word1.encode('utf-8')) and w2v.has_key(word2.encode('utf-8')):
        a = map(float, w2v[word1.encode('utf-8')])
        b = map(float, w2v[word2.encode('utf-8')])
        a = np.array(a).reshape(1,-1)
        b = np.array(b).reshape(1, -1)
    else:
        return 0.1# if 0 no effect on !=

    cos_sim = cosine_similarity(a,b)
     #print word1, word2, cos_sim
    return cos_sim

def calJaccard(str1, str2, aa):
    # jieba
    jieba.load_userdict("data/entities")


    str1_list = [x for x in jieba.cut(str1)]
    str2_list = [x for x in jieba.cut(str2)]
    str1_list_no_stop = []
    str2_list_no_stop = []
    for word in str1_list:
        if word.encode("utf-8") not in stopwords:
            if word != '' and len(word) != 0:  str1_list_no_stop.append(word)

    for word in str2_list:
        if word.encode("utf-8") not in stopwords:
            if word != '' and len(word) != 0:   str2_list_no_stop.append(word)

    if len(str2_list_no_stop) <= 0 or len(str1_list_no_stop) == 0: # not null
         str1_list_no_stop = str1_list
         str2_list_no_stop = str2_list

    len1 = len(str1_list_no_stop)
    len2 = len(str2_list_no_stop)

    # martix
    martix = [[0 for col in range(len2)] for row in range(len1)]
    #
    for i in range(0, len1):
        for j in range(0, len2):
            sim = martix[i][j] = calSim(str1_list_no_stop[i], str2_list_no_stop[j])
            # print i, j, str1_list[i], str2_list[j], sim
    # fenzi
    a = aa
    mmax = a
    pos = [0, 0]
    total = 0
    while (mmax >= a):
        mmax = 0
        for i in range(len1):
            for j in range(len2):
                if (martix[i][j] > mmax):
                    mmax = martix[i][j]
                    pos = [i, j]
        if (mmax >= a):
            total += mmax;
        for i in range(len1):
            martix[i][pos[1]] = 0
        for j in range(len2):
            martix[pos[0]][j] = 0
    # fenmu
    m = 0
    diff = 0
    for i in range(len1):
        for j in range(len2):
            if (martix[i][j] != 0):
                m += 1
                diff += (1 - martix[i][j])
    fenmu = total + m * diff + 0.01# not 0

    # print fenmu, total
    return total / fenmu

# main


w2v = {}
print "read w2v ..."
with open("data/wiki.zh.vec", "r") as ins:
    for line in ins:
        word = line.split()[0]
        vec = line.split()[1:]
        w2v[word] = vec


data = pd.read_csv("data/atec_nlp_sim_train_split.csv", sep="\t", names=['id','str1','str2','out'])

stopwords = []
with open("data/stopwords", "r") as ins:
    for line in ins:
        stopwords.append(line.strip())


print data["str1"].size
for aa in [0.98]:
    zero_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    one_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    with open("result/a=" +str(aa)+ ".txt", 'w') as fout:
        for i in range(data["str1"].size):
            result = calJaccard(data["str1"][i].strip() , data["str2"][i].strip() , aa)
            #print data["str1"][i], data["str2"][i], data["out"][i], result
            if (i % 100 == 0): print i, data["str1"][i], data["str2"][i], data["out"][i], result
            fout.write(data["str1"][i]+" "+ data["str2"][i] + " " + str(data["out"][i]) + str(result) + "\r\n")

            if (data["out"][i] == 0):

                if (result >= 0 and result < 0.1):              zero_count[0] += 1
                if (result >= 0.1 and result < 0.2):            zero_count[1] += 1
                if (result >= 0.2 and result < 0.3):            zero_count[2] += 1
                if (result >= 0.3 and result < 0.4):            zero_count[3] += 1
                if (result >= 0.4 and result < 0.5):            zero_count[4] += 1
                if (result >= 0.5 and result < 0.6):            zero_count[5] += 1
                if (result >= 0.6 and result < 0.7):            zero_count[6] += 1
                if (result >= 0.7 and result < 0.8):            zero_count[7] += 1
                if (result >= 0.8 and result < 0.9):            zero_count[8] += 1
                if (result >= 0.9 and result < 1):              zero_count[9] += 1

            if (data["out"][i] == 1):
                if (result >= 0 and result < 0.1):              one_count[0] += 1
                if (result >= 0.1 and result < 0.2):            one_count[1] += 1
                if (result >= 0.2 and result < 0.3):            one_count[2] += 1
                if (result >= 0.3 and result < 0.4):            one_count[3] += 1
                if (result >= 0.4 and result < 0.5):            one_count[4] += 1
                if (result >= 0.5 and result < 0.6):            one_count[5] += 1
                if (result >= 0.6 and result < 0.7):            one_count[6] += 1
                if (result >= 0.7 and result < 0.8):            one_count[7] += 1
                if (result >= 0.8 and result < 0.9):            one_count[8] += 1
                if (result >= 0.9 and result <= 1):             one_count[9] += 1
        fout.write("-------------------\n")

        for i in zero_count:
            fout.write(str(i) + ',')  # \r\n为换行符
        fout.write('\r\n')  # \r\n为换行符
        for i in one_count:
            fout.write(str(i) + ',')  # \r\n为换行符.

        # cal
        FN = sum(one_count[5:])
        TP = sum(one_count[:5])

        FP = sum(zero_count[5:])
        TN = sum(zero_count[:5])
        print FN,TP
        print FP,TN

        ACC = float(TP+TN) / float(TP+TN+FP+FN)
        RECALL = float(TP) / float(TP+FN)
        PRECISION = float(TP) / float(TP+FP)
        F1 = float(2*TP) / float(2*TP + FN + FP)
        F1X= 2*PRECISION*RECALL/(PRECISION+RECALL)

        print TP, FP, TN, FN
        print F1, F1X







