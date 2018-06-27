#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
TextCNN: classification
"""
import os
import re
import time
import jieba
import argparse
import numpy as np
from collections import Counter
from datetime import timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset



base_dir = '../data'
entity_file = os.path.join(base_dir, 'entities')
train_file = os.path.join(base_dir, 'atec_nlp_sim_train_split.csv')
vocab_file = os.path.join(base_dir, 'atec_nlp_sim_train_split.vocab.txt')

save_path = 'modelSave'  # model save path
if not os.path.exists(save_path):
    os.mkdir(save_path)
model_file = os.path.join(save_path, 'TextCNN_Classification.pt')


parser = argparse.ArgumentParser(description='LSTM text classification')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate [default: 0.001]')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs for train')
parser.add_argument('--batch-size', type=int, default=16,
                    help='batch size for training [default: 16]')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda-able', action='store_true',
                    help='enables cuda')

parser.add_argument('--save', type=str, default='./LSTM_Text.pt',
                    help='path to save the final model')
parser.add_argument('--data', type=str, default='./data/corpus.pt',
                    help='location of the data corpus')

parser.add_argument('--dropout', type=float, default=0.5,
                    help='the probability for dropout (0 = no dropout) [default: 0.5]')
parser.add_argument('--embed-dim', type=int, default=64,
                    help='number of embedding dimension [default: 64]')
parser.add_argument('--hidden-size', type=int, default=128,
                    help='number of lstm hidden dimension [default: 128]')
parser.add_argument('--lstm-layers', type=int, default=3,
                    help='biLSTM layer numbers')
parser.add_argument('--bidirectional', action='store_true',
                    help='If True, becomes a bidirectional LSTM [default: False]')

args = parser.parse_args()
args.vocab_size = 5000
args.label_size = 2
torch.manual_seed(args.seed)

def open_file(filename, mode='r'):
    """
    Commonly used file reader and writer, change this to switch between python2 and python3.
    :param filename: filename
    :param mode: 'r' and 'w' for read and write respectively
    """
    return open(filename, mode)

class LSTMConfig(object):
    """
    LSTM Parameters
    """
    vocab_size = 8000  # most common words
    seq_length = 50  # maximum length of sequence
    dev_split = 0.1  # percentage of dev data

    embedding_dim = 64  # embedding vector size
    num_filters = 100  # number of the convolution filters (feature maps)
    lstm_layers = 3

    hidden_dim = 128  # hidden size of fully connected layer

    dropout_prob = 0.5  # how much probability to be dropped
    learning_rate = 1e-3  # learning rate
    batch_size = 16  # batch size for training
    num_epochs = 100  # total number of epochs

    num_classes = 2  # number of classes

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, input):
        mu = torch.mean(input, dim=-1, keepdim=True)
        sigma = torch.std(input, dim=-1, keepdim=True).clamp(min=self.eps)
        output = (input - mu) / sigma
        return output * self.weight.expand_as(output) + self.bias.expand_as(output)


class LSTM_Text(nn.Module):
    def __init__(self, args):
        super(LSTM_Text, self).__init__()
        for k, v in args.__dict__.items():
            self.__setattr__(k, v)

        self.num_directions = 2 if self.bidirectional else 1

        self.lookup_table = nn.Embedding(self.vocab_size, self.embed_dim)

        self.lstm = nn.LSTM(self.embed_dim,
                            self.hidden_size,
                            self.lstm_layers,
                            dropout=self.dropout,
                            bidirectional=self.bidirectional)
        self.ln = LayerNorm(self.hidden_size * self.num_directions)
        self.logistic = nn.Linear(self.hidden_size * self.num_directions,
                                  self.label_size)

        self._init_weights()

    def _init_weights(self, scope=1.):
        self.lookup_table.weight.data.uniform_(-scope, scope)
        self.logistic.weight.data.uniform_(-scope, scope)
        self.logistic.bias.data.fill_(0)

    def init_hidden(self):
        num_layers = self.lstm_layers * self.num_directions

        weight = next(self.parameters()).data
        return (Variable(weight.new(num_layers, self.batch_size, self.hidden_size).zero_()), Variable(weight.new(num_layers, self.batch_size, self.hidden_size).zero_()))

    def forward(self, input, hidden):
        encode = self.lookup_table(input)
        lstm_out, hidden = self.lstm(encode.transpose(0, 1), hidden)
        output = self.ln(lstm_out)[-1]
        return F.log_softmax(self.logistic(output)), hidden

class Corpus(object):
    def __init__(self, filepath, vocab_file, dev_split=0.1, max_length=50, vocab_size=8000):
        print "Corpus_init"
        print clean_str("	我想用蚂蚁借呗怎么用不了	蚂蚁借呗设置了一次性还款，现在想分期还款，怎么操作	0")
        # loading data
        lines = [s.strip().split("\t") for s in open_file(filepath)]
        id = [x[0] for x in lines]
        #sen1 = [x[1] for x in lines]
        #sen2 = [x[2] for x in lines]
        total_sen = [x[1]+x[2]  for x in lines]
        #print total_sen
        tag = [int(x[3]) for x in lines]

        if not os.path.exists(vocab_file):
            build_vocab(total_sen, vocab_file, vocab_size)

        self.words, self.word_to_id = read_vocab(vocab_file)

        for i in range(len(total_sen)):  # tokenizing and padding
            total_sen[i] = process_text(total_sen[i], self.word_to_id, max_length, clean=False)

        x_data = np.array(total_sen)
        y_data = np.array(tag)

        # shuffle
        indices = np.random.permutation(np.arange(len(x_data)))
        x_data = x_data[indices]
        y_data = y_data[indices]

        # train/dev split
        num_train = int((1 - dev_split) * len(x_data))
        self.x_train = x_data[:num_train]
        self.y_train = y_data[:num_train]
        self.x_test = x_data[num_train:]
        self.y_test = y_data[num_train:]


    def __str__(self):
        return 'Training: {}, Testing: {}, Vocabulary: {}'.format(len(self.x_train), len(self.x_test), len(self.words))

def process_text(text, word_to_id, max_length, clean=True):
    """tokenizing and padding"""
    if clean:  # if the data needs to be cleaned
        text = clean_str(text)
    text = jieba.cut(text)

    text = [word_to_id[x.encode("utf-8")] for x in text if x.encode("utf-8") in word_to_id]
    if len(text) < max_length:
        text = [0] * (max_length - len(text)) + text
    return text[:max_length]

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
        all_data.extend(jieba.cut(content))

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

def loadData(path):
    print "loadData"

def get_time_dif(start_time):
    """
    Return the time used since start_time.
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def evaluate(data, model, loss):
    """
    Evaluation, return accuracy and loss
    """
    model.eval()  # set mode to evaluation to disable dropout
    data_loader = DataLoader(data, batch_size=50)

    data_len = len(data)
    total_loss = 0.0
    y_true, y_pred = [], []

    for data, label in data_loader:
        data, label = Variable(data), Variable(label)

        output = model(data)
        losses = loss(output, label)

        total_loss += losses.data
        pred = torch.max(output.data, dim=1)[1].cpu().numpy().tolist()
        y_pred.extend(pred)
        y_true.extend(label.data)

    acc = (np.array(y_true) == np.array(y_pred)).sum()
    return float(acc) / data_len, total_loss / data_len

def repackage_hidden(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def train():
    print "train"
    start_time = time.time()
    config = LSTMConfig()
    corpus = Corpus(train_file, vocab_file, config.dev_split, config.seq_length, config.vocab_size)
    print(corpus)
    config.vocab_size = len(corpus.words)

    train_data = TensorDataset(torch.LongTensor(corpus.x_train), torch.LongTensor(corpus.y_train))
    test_data = TensorDataset(torch.LongTensor(corpus.x_test), torch.LongTensor(corpus.y_test))

    print('Configuring CNN model...')
    model = LSTM_Text(args)
    print(model)

    # optimizer and loss function
    criterion = nn.CrossEntropyLoss(size_average=False)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    hidden = model.init_hidden()
    # set the mode to train
    print("Training and evaluating...")
    best_acc = 0.0

    for epoch in range(config.num_epochs):
        # load the training data in batch
        model.train()
        train_loader = DataLoader(train_data, batch_size=config.batch_size)
        for x_batch, y_batch in train_loader:
            inputs, targets = Variable(x_batch), Variable(y_batch)

            optimizer.zero_grad()
            hidden = repackage_hidden(hidden)
            outputs = model(inputs, hidden)  # forward computation
            loss = criterion(outputs, targets)

            # backward propagation and update parameters
            loss.backward()
            optimizer.step()

        # evaluate on both training and test dataset
        train_acc, train_loss = evaluate(train_data, model, criterion)
        test_acc, test_loss = evaluate(test_data, model, criterion)

        if test_acc > best_acc:
            # store the best result
            best_acc = test_acc
            improved_str = '*'
            torch.save(model.state_dict(), model_file)
        else:
            improved_str = ''

        time_dif = get_time_dif(start_time)
        msg = "Epoch {0:3}, Train_loss: {1:>7.2}, Train_acc {2:>6.2%}, " \
              + "Test_loss: {3:>6.2}, Test_acc {4:>6.2%}, Time: {5} {6}"
        print(msg.format(epoch + 1, train_loss, train_acc, test_loss, test_acc, time_dif,  improved_str))

train()
