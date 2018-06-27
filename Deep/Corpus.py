#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import re
import time
import jieba
import numpy as np
from collections import Counter
from datetime import timedelta

base_dir = '../data'
entity_file = os.path.join(base_dir, 'entities')
train_file = os.path.join(base_dir, 'atec_nlp_sim_train_split.csv')
#valid_file  = os.path.join(base_dir, 'atec_nlp_sim_train_split.csv')
test_file  = os.path.join(base_dir, 'atec_nlp_sim_test_split.csv')
vocab_file = os.path.join(base_dir, 'atec_nlp_sim_train.vocab.txt')
result_file= os.path.join("result_SiameseCNN.txt")

save_path = 'modelSave'  # model save path
if not os.path.exists(save_path):
    os.mkdir(save_path)
model_file = os.path.join(save_path, 'SiameseTextCNN.pt')


class Corpus(object):
    def __init__(self, filepath, vocab_file, dev_split=0.1, max_length=50, vocab_size=8000):
        print "Corpus_init"
        # loading data
        lines = [s.strip().split("\t") for s in open_file(filepath)]
        id = [x[0] for x in lines]
        sen1 = [x[1] for x in lines]
        sen2 = [x[2] for x in lines]
        total_sen = [x[1] + x[2] for x in lines]

        tag = [int(x[3]) for x in lines]

        if not os.path.exists(vocab_file):
            build_vocab(total_sen, vocab_file, vocab_size)

        self.words, self.word_to_id = read_vocab(vocab_file)

        for i in range(len(sen1)):  # tokenizing and padding
            sen1[i] = process_text(sen1[i], self.word_to_id, max_length, clean=False)
        for i in range(len(sen2)):  # tokenizing and padding
            sen2[i] = process_text(sen2[i], self.word_to_id, max_length, clean=False)

        x_data1 = np.array(sen1)
        x_data2 = np.array(sen2)
        y_data = np.array(tag)

        # shuffle
        # indices = np.random.permutation(np.arange(len(y_data)))
        # x_data1 = x_data1[indices]
        # x_data2 = x_data2[indices]
        # y_data = y_data[indices]

        # train/dev split
        num_train = int((1 - dev_split) * len(y_data))
        self.x_train1 = x_data1[:num_train]
        self.x_train2 = x_data2[:num_train]
        self.x_test1 = x_data1[num_train:]
        self.x_test2 = x_data2[num_train:]

        self.y_train = y_data[:num_train]
        self.y_test = y_data[num_train:]

    def __str__(self):
        return 'Training: {}, Testing: {}, Vocabulary: {}'.format(len(self.y_train), len(self.y_test), len(self.words))


def process_text(text, word_to_id, max_length, clean=True):
    """tokenizing and padding"""
    if clean:  # if the data needs to be cleaned
        text = clean_str(text)
    jieba.load_userdict(entity_file)
    text = jieba.cut(text,cut_all=True)

    text = [word_to_id[x.encode("utf-8")] for x in text if x.encode("utf-8") in word_to_id]
    if len(text) < max_length:
        text = [0] * (max_length - len(text)) + text
    return text[:max_length]


def open_file(filename, mode='r'):
    """
    Commonly used file reader and writer, change this to switch between python2 and python3.
    :param filename: filename
    :param mode: 'r' and 'w' for read and write respectively
    """
    return open(filename, mode)


def read_vocab(vocab_file):
    """
    Read vocabulary from file.
    """
    words = open_file(vocab_file).read().strip().split('\n')
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def build_vocab(data, vocab_dir, vocab_size=8000):
    """
    Build vocabulary file from training data.
    """
    print('Building vocabulary...')
    jieba.load_userdict(entity_file)
    all_data = []  # group all data
    for content in data:
        all_data.extend(jieba.cut(content, cut_all=True))

    counter = Counter(all_data)  # count and get the most common words
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))

    words = ['<PAD>'] + list(words)  # add a padding with id 0 to pad the sentence to same length
    # with open(vocab_dir, 'w') as fout:
    #     for word in words:
    #         fout.write(word.encode('utf-8')+'\n')
    open_file(vocab_dir, 'w').write('\n'.join(words).encode('utf-8') + '\n')


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def get_time_dif(start_time):
    """
    Return the time used since start_time.
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))