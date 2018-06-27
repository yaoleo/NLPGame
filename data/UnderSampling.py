#!/usr/bin/python
# -*- coding: utf-8 -*-

def process():
    print "ss"
    inpath1 = "atec_nlp_sim_train.csv"
    inpath2 = "atec_nlp_sim_train_add.csv"
    outpath = "atec_nlp_sim_undersample.csv"
    with open(inpath1, 'r') as fin1,open(inpath2) as fin2,open(outpath, 'w') as fout:
        total_num = 0
        pos_num = 0
        neg_num = 0
        flag = 0
        print "s"
        for line in fin1:

            lineno, sen1, sen2, tag = line.strip().split('\t')
            if tag == "1":
                total_num += 1
                pos_num += 1
                fout.write(line)
                flag += 1
            if tag == "0" and flag > 0:
                total_num += 1
                fout.write(line)
                flag -= 1
        for line in fin2:
            lineno, sen1, sen2, tag = line.strip().split('\t')
            if tag == "1":
                total_num += 1
                pos_num += 1
                fout.write(line)
                flag += 1
            if tag == "0" and flag > 0:
                total_num += 1
                fout.write(line)
                flag -= 1
        print pos_num, total_num
print "underSampling..."
process()
print "Finsish"