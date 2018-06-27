#/usr/bin/env python
#coding=utf-8
import jieba
import sys
DATAPATH  = "../../data/"

def process(inpath, outpath):
    with open(inpath, 'r') as fin, open(outpath, 'w') as fout:
        TP, FP, TN, FN = 0, 0, 0, 0
        for line in fin:
            lineno, sen1, sen2,tag = line.strip().split('\t')
            words1= [ w for w in jieba.cut(sen1) if w.strip() ]
            words2= [ w for w in jieba.cut(sen2) if w.strip() ]
            union = words1 + words2
            same_num = 0
            for w in union:
                if w in words1 and w in words2:
                    same_num += 1
            if same_num * 2 >= len(union):
                fout.write(lineno + '\t1\n')
            else:
                fout.write(lineno + '\t0\n')

            if same_num * 2 >= len(union) and tag == "1":  #
                TP += 1
            if same_num * 2 < len(union) and tag == "0":
                TN += 1
            if same_num * 2 >= len(union) and tag == "0":
                FP += 1
            if same_num * 2 < len(union) and tag == "1":
                FN += 1

        PRECISION = float(TP) / float(TP + FP)
        RECALL = float(TP) / float(TP + FN)

        ACC = float(TP + TN) / float(TP + TN + FP + FN)
        F1 = float(2 * TP) / float(2 * TP + FN + FP)
        F1X = 2 * PRECISION * RECALL / (PRECISION + RECALL)
        # print TP, FP, TN, FN

        print F1, ACC, PRECISION, RECALL, TP, FN + FP

if __name__ == '__main__':
    process(DATAPATH +sys.argv[1], sys.argv[2])
