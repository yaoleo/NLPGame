# /usr/bin/env python
# coding=utf-8
import jieba
import sys
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

DATAPATH = "../../data/"
HYPERPARA_cc = [0.3]

def calSim(word1, word2, bb = 0.5):

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

def calAllWordW2VSum(str1,str2):
    # jieba
    jieba.load_userdict(DATAPATH + "entities")

    str1_list = [x for x in jieba.cut(str1, cut_all=True)]
    str2_list = [x for x in jieba.cut(str2, cut_all=True)]
    str1_list_no_stop = []
    str2_list_no_stop = []
    for word in str1_list:
        if word.encode("utf-8") not in stopwords:
            if len(word) != 0 and word.encode("utf-8") != "花呗" and word.encode(
                    "utf-8") != "借呗":  str1_list_no_stop.append(word)

    for word in str2_list:
        if word.encode("utf-8") not in stopwords:
            if len(word) != 0 and word.encode("utf-8") != "花呗" and word.encode(
                    "utf-8") != "借呗":   str2_list_no_stop.append(word)

    if len(str2_list_no_stop) <= 0 or len(str1_list_no_stop) == 0: # not null
         str1_list_no_stop = str1_list
         str2_list_no_stop = str2_list

    len1 = len(str1_list_no_stop)
    len2 = len(str2_list_no_stop)
    sim = 0;
    for word1 in str1_list_no_stop:
        for word2 in str2_list_no_stop:
            sim += calSim(word1,word2)
    return sim/float(len1*len2)

def calSumWordW2V(str1, str2):
    # jieba
    jieba.load_userdict(DATAPATH + "entities")

    str1_list = [x for x in jieba.cut(str1, cut_all=True)]
    str2_list = [x for x in jieba.cut(str2, cut_all=True)]
    str1_list_no_stop = []
    str2_list_no_stop = []
    for word in str1_list:
        if word.encode("utf-8") not in stopwords:
            if len(word) != 0 and word.encode("utf-8") != "花呗" and word.encode(
                "utf-8") != "借呗":  str1_list_no_stop.append(word)

    for word in str2_list:
        if word.encode("utf-8") not in stopwords:
            if len(word) != 0 and word.encode("utf-8") != "花呗" and word.encode(
                "utf-8") != "借呗":   str2_list_no_stop.append(word)

    if len(str2_list_no_stop) <= 0 or len(str1_list_no_stop) == 0:  # not null
        str1_list_no_stop = str1_list
        str2_list_no_stop = str2_list

    vec1 = np.array([0] * 300).reshape(1, -1)
    vec2 = np.array([0] * 300).reshape(1, -1)
    for word in str1_list_no_stop:
        if w2v.has_key(word.encode('utf-8')):
            vec1 = np.array(map(float, w2v[word.encode('utf-8')])).reshape(1, -1)

    for word in str2_list_no_stop:
        if w2v.has_key(word.encode('utf-8')):
            vec2 = np.array(map(float, w2v[word.encode('utf-8')])).reshape(1, -1)

    len1 = len(str1_list_no_stop)
    len2 = len(str2_list_no_stop)
    cos_sim = cosine_similarity(vec1/len1, vec2/len2)
    return cos_sim

def process(inpath, outpath):
    # read data and process
    print "processing..."
    for cc in HYPERPARA_cc:
        #########################
        TP, FP, TN, FN = 0, 0, 0, 0
        # PRECISION, RECALL, ACCURACY, F1 = 0
        with open(inpath, 'r') as fin, open(outpath, 'w') as fout:
            for line in fin:
                lineno, sen1, sen2, tag = line.strip().split('\t')
                result = calAllWordW2VSum(sen1, sen2)
                # print lineno, sen1, sen2, tag, result

                if result >= cc and tag == "1":  #
                    TP += 1
                if result < cc and tag == "0":
                    TN += 1
                if result >= cc and tag == "0":
                    FP += 1
                if result < cc and tag == "1":
                    FN += 1

        PRECISION = float(TP) / float(TP + FP)
        RECALL = float(TP) / float(TP + FN)

        ACC = float(TP + TN) / float(TP + TN + FP + FN)
        F1 = float(2 * TP) / float(2 * TP + FN + FP)
        F1X = 2 * PRECISION * RECALL / (PRECISION + RECALL)
        # print TP, FP, TN, FN

        print F1, ACC, PRECISION, RECALL, TP, FN + FP

        ############################


if __name__ == '__main__':
    # read w2v
    w2v = {}
    print "loading w2v ..."
    with open(DATAPATH + "wiki.zh.vec", "r") as ins:
        for line in ins:
            word = line.split()[0]
            vec = line.split()[1:]
            w2v[word] = vec
    print "loading stopwords ..."
    # read stopwords
    stopwords = []
    with open(DATAPATH + "stopwords", "r") as ins:
        for line in ins:
            stopwords.append(line.strip())
    process(DATAPATH+sys.argv[1], sys.argv[2])
