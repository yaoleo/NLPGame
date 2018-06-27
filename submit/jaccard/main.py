#/usr/bin/env python
#coding=utf-8
import jieba
import sys
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
DATAPATH  = "../../data/"
HYPERPARA_aa = [0.96]
HYPERPARA_bb = [0.5]
HYPERPARA_cc = [0.3]

def calSim(word1, word2, bb):

    if w2v.has_key(word1.encode('utf-8')) and w2v.has_key(word2.encode('utf-8')):
        a = map(float, w2v[word1.encode('utf-8')])
        b = map(float, w2v[word2.encode('utf-8')])
        a = np.array(a).reshape(1,-1)
        b = np.array(b).reshape(1, -1)
    else:
        return bb# if 0 no effect on !=

    cos_sim = cosine_similarity(a,b)
     #print word1, word2, cos_sim
    return cos_sim

def calJaccard(str1, str2, aa, bb):
    # jieba
    jieba.load_userdict(DATAPATH+"entities")

    str1_list = [x for x in jieba.cut(str1, cut_all=True)]
    str2_list = [x for x in jieba.cut(str2, cut_all=True)]
    str1_list_no_stop = []
    str2_list_no_stop = []
    for word in str1_list:
        if word.encode("utf-8") not in stopwords:
            if len(word) != 0 and word.encode("utf-8") != "花呗" and word.encode("utf-8") != "借呗":  str1_list_no_stop.append(word)

    for word in str2_list:
        if word.encode("utf-8") not in stopwords:
            if len(word) != 0 and word.encode("utf-8") != "花呗" and word.encode("utf-8") != "借呗":   str2_list_no_stop.append(word)

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
            sim = martix[i][j] = calSim(str1_list_no_stop[i], str2_list_no_stop[j], bb)
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

    # print total / fenmu
    return total / fenmu

def process(inpath, outpath):
    # read data and process
    print "processing..."
    for aa in HYPERPARA_aa:
        for bb in HYPERPARA_bb:
            for cc in HYPERPARA_cc:
                #########################
                TP, FP, TN, FN = 0,0,0,0
                # PRECISION, RECALL, ACCURACY, F1 = 0
                with open(inpath, 'r') as fin, open(outpath, 'w') as fout:
                    for line in fin:
                        lineno, sen1, sen2, tag = line.strip().split('\t')

                        result = calJaccard(sen1, sen2, aa, bb)
                        #print lineno, sen1, sen2, tag, result
                        if result >= cc:
                            fout.write(lineno + '\t1\n')
                        else:
                            fout.write(lineno + '\t0\n')

                        if result >= cc and tag == "1": #
                            TP += 1
                        if result < cc and tag  == "0":
                            TN += 1
                        if result >= cc and tag == "0":
                            FP += 1
                        if result < cc and tag == "1":
                            FN += 1

                PRECISION = float(TP) / float(TP+FP)
                RECALL = float(TP) / float(TP + FN)

                ACC = float(TP + TN) / float(TP + TN + FP + FN)
                F1 = float(2 * TP) / float(2 * TP + FN + FP)
                F1X = 2 * PRECISION * RECALL / (PRECISION + RECALL)
                #print TP, FP, TN, FN
                print "aa_juzhen:", aa,"bb_weidengluci:", bb, "cc_shuchufenlei:", cc
                print F1, ACC, PRECISION, RECALL, TP, FN+FP

                ############################

if __name__ == '__main__':
    # read w2v
    w2v = {}
    print "loading w2v ..."
    with open(DATAPATH+"wiki.zh.vec", "r") as ins:
        for line in ins:
            word = line.split()[0]
            vec = line.split()[1:]
            w2v[word] = vec
    print "loading stopwords ..."
    # read stopwords
    stopwords = []
    with open(DATAPATH+"stopwords", "r") as ins:
        for line in ins:
            stopwords.append(line.strip())
    process(sys.argv[1], sys.argv[2])
