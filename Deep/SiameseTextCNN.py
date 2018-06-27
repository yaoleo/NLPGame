#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
把两个CNN得到的句向量接起来当作二分类问题处理
"""
from Corpus import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

class TCNNConfig(object):
    """
    CNN Parameters
    """
    vocab_size = 8000  # most common words
    seq_length = 50  # maximum length of sequence
    dev_split = 0.2  # percentage of dev data

    embedding_dim = 128  # embedding vector size
    num_filters = 100  # number of the convolution filters (feature maps)
    kernel_sizes = [3,4,5]  # three kind of kernels (windows)

    hidden_dim = 128  # hidden size of fully connected layer

    dropout_prob = 0.5  # how much probability to be dropped
    learning_rate = 1e-3  # learning rate
    batch_size = 50  # batch size for training
    num_epochs = 20  # total number of epochs

    num_classes = 2  # number of classes


class TextCNN(nn.Module):
    """
    CNN text classification model, based on the paper.
    """

    def __init__(self, config):
        super(TextCNN, self).__init__()

        V = config.vocab_size
        E = config.embedding_dim

        Nf = config.num_filters
        Ks = config.kernel_sizes

        Dr = config.dropout_prob
        C = config.num_classes
##########################################
        self.embedding = nn.Embedding(V, E)  # embedding layer

        # three different convolutional layers
        self.convs = nn.ModuleList([nn.Conv1d(E, Nf, k) for k in Ks])
        self.dropout = nn.Dropout(Dr)  # a dropout layer
        #self.fc1 = nn.Linear(2*len(Ks)* Nf, 1)  # a dense layer for classification
        self.fc1 = nn.Linear(2*len(Ks) * Nf, C)  # a dense layer for classification
        #self.fc2 = nn.Linear(2*3*Nf, len(Ks) * Nf)
###############################################



    @staticmethod
    def conv_and_max_pool(x, conv):
        """Convolution and global max pooling layer"""
        return F.relu(conv(x).permute(0, 2, 1).max(1)[0])

    def forward_once(self, inputs):

        x = [self.conv_and_max_pool(inputs, k) for k in self.convs]  # convolution and global max pooling
        output = torch.cat(x, 1)
        return output

    def forward(self, sen1, sen2):
        # Conv1d takes in (batch, channels, seq_len), but raw embedded is (batch, seq_len, channels)
        embedded1 = self.embedding(sen1).permute(0, 2, 1)
        embedded2 = self.embedding(sen2).permute(0, 2, 1)
        output1 = self.forward_once(embedded1)
        output2 = self.forward_once(embedded2)

        output = torch.cat((output1,output2),1)
        output = self.fc1(self.dropout(output))  # concatenation and dropout

        return output


def evaluate(data, model, loss):
    """
    Evaluation, return accuracy and loss
    """
    model.eval()  # set mode to evaluation to disable dropout
    data_loader = DataLoader(data, batch_size=50)

    data_len = len(data)
    total_loss = 0.0
    prelabel = []
    TP, FP, TN, FN = 0, 0, 0, 0
    pos_num, neg_num = 0, 0
    for data1, data2, label in data_loader:

        data1, data2, label = Variable(data1),Variable(data2), Variable(label)
        label_ones = int(label.sum().data)


        output = model(data1,data2)
        pred = torch.max(output.data, dim=1)[1].cpu().numpy().tolist()

        cc = 0.1

        for i in range(len(label)):
            if label[i] == 1: pos_num += 1
            else: neg_num += 1

        for i in range(len(pred)):
            if pred[i] == 1 and label[i] == 1:  #
                TP += 1
            if pred[i] == 0 and label[i] == 0:
                TN += 1
            if pred[i] == 1 and label[i] == 0:
                FP += 1
            if pred[i] == 0 and label[i] == 1:
                FN += 1

        losses = loss(output, label)
        total_loss += losses.data

    PRECISION = float(TP) / float(TP + FP)
    RECALL = float(TP) / float(TP + FN)

    ACC = float(TP + TN) / float(TP + TN + FP + FN)
    F1 = float(2 * TP) / float(2 * TP + FN + FP)
    print pos_num,neg_num
    print TP,TN,FP,FN
    print "ACC,F1:",ACC,F1
    return float(total_loss) / data_len, F1


def train():
    print "train"
    start_time = time.time()
    config = TCNNConfig()
    corpus = Corpus(train_file, vocab_file, 0.0, config.seq_length, config.vocab_size)
    testcorpus = Corpus(test_file,vocab_file, 1.0, config.seq_length, config.vocab_size)
    print(corpus)
    print(testcorpus)
    config.vocab_size = len(corpus.words)

    train_data = TensorDataset(torch.LongTensor(corpus.x_train1), torch.LongTensor(corpus.x_train2),
                               torch.LongTensor(corpus.y_train))
    validation_data = TensorDataset(torch.LongTensor(corpus.x_test1), torch.LongTensor(corpus.x_test2),
                              torch.LongTensor(corpus.y_test))

    test_data = TensorDataset(torch.LongTensor(testcorpus.x_test1), torch.LongTensor(testcorpus.x_test2),
                              torch.LongTensor(testcorpus.y_test))

    print('Configuring CNN model...')
    model = TextCNN(config)
    print(model)

    # optimizer and loss function
    # criterion = nn.CrossEntropyLoss(size_average=False)
    # class_weight = Variable(torch.FloatTensor([1, 4]))  # 这里正例比较少，因此权重要大一些
    # target = Variable(torch.FloatTensor(50).random_(2))
    # weight = class_weight[target.long()]  # (3, 4)
    # criterion = torch.nn.BCELoss(weight=weight, reduce=False, size_average=False)

    #criterion = torch.nn.BCELoss(reduce=False, size_average=False)
    # weight = torch.Tensor([1, 4])
    criterion = nn.CrossEntropyLoss(size_average=False)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # set the mode to train
    print("Training and evaluating...")
    best_F1 = 0.0
    for epoch in range(config.num_epochs):
        # load the training data in batch
        model.train()
        train_loader = DataLoader(train_data, batch_size=config.batch_size)
        batch = 0
        for x1_batch, x2_batch, y_batch in train_loader:
            batch+=1
            if batch%100 == 0: print epoch,batch
            inputs1, inputs2, targets = Variable(x1_batch), Variable(x2_batch), Variable(y_batch)

            optimizer.zero_grad()
            outputs = model(inputs1, inputs2)  # forward computation

            loss = criterion(outputs, targets)
            """
            todo
            """
            # backward propagation and update parameters
            loss.backward()
            optimizer.step()

        # evaluate on both training and test dataset

        train_loss, train_F1 = evaluate(train_data, model, criterion)
        #validation_loss, validation_F1  = evaluate(validation_data, model, criterion)
        test_loss, test_F1 = evaluate(test_data, model, criterion)


        print "train_loss:",train_loss

        if test_F1 > best_F1:
             # store the best result
             best_F1 = test_F1
             improved_str = '*'
             torch.save(model.state_dict(), model_file)
        else:
             improved_str = ''

        time_dif = get_time_dif(start_time)
        msg = "Epoch {0:3}, Train_loss: {1:>7.2}, Train_F1 {2:>6.2%}, " \
               + "Test_loss: {3:>6.2}, Test_F1 {4:>6.2%}, Time: {5} {6}"
        print(msg.format(epoch + 1, train_loss, train_F1, test_loss, test_F1, time_dif,  improved_str))


def predict(sen1, sen2):
    # load config and vocabulary
    config = TCNNConfig()
    _, word_to_id = read_vocab(vocab_file)

    # load model
    model = TextCNN(config)
    model.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))

    # process text
    sen1 = process_text(sen1, word_to_id, config.seq_length)
    sen1 = Variable(torch.LongTensor([sen1]), volatile=True)

    sen2 = process_text(sen2, word_to_id, config.seq_length)
    sen2 = Variable(torch.LongTensor([sen2]), volatile=True)

    model.eval()  # very important
    output = model(sen1,sen2)

    return output.data


train()


TP, FP, TN, FN = 0,0,0,0
cc = 0.2

with open(test_file, 'r') as fin, open(result_file, 'w') as fout:
    for line in fin:
        lineno, sen1, sen2, tag = line.strip().split('\t')

        result = predict(sen1, sen2)
        # print lineno, sen1, sen2, tag, result
        if result >= cc:
            fout.write(lineno + '\t1\n')
        else:
            fout.write(lineno + '\t0\n')

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
    print "pre", cc
    print F1, ACC, PRECISION, RECALL, TP, FN + FP

