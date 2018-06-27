#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
TextCNN: cal sim between two sentences
"""

from Corpus import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset


class SiameseTCNNConfig(object):
    """
    CNN Parameters
    """
    vocab_size = 8000  # most common words
    seq_length = 50  # maximum length of sequence
    dev_split = 0.1  # percentage of dev data

    embedding_dim = 300  # embedding vector size
    num_filters = 100  # number of the convolution filters (feature maps)
    kernel_sizes = [3, 4, 5]  # three kind of kernels (windows)

    hidden_dim = 128  # hidden size of fully connected layer

    dropout_prob = 0.5  # how much probability to be dropped
    learning_rate = 0.001  # learning rate
    batch_size = 50  # batch size for training
    num_epochs = 20  # total number of epochs

    contra_loss = True


class SiameseTextCNN(nn.Module):
    """
    CNN text classification model, based on the paper.
    """
    def __init__(self, config):
        super(SiameseTextCNN, self).__init__()

        vocab_size = config.vocab_size
        embedding_dim = config.embedding_dim

        num_filters = config.num_filters
        filter_sizes = config.kernel_sizes
        max_sent_len = config.seq_length
        self.dropout_prob = config.dropout_prob
        self.contra_loss = True
        ##########################################
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)  # embedding layer

        conv_blocks = []

        for filter_size in filter_sizes:
            maxpool_kernel_size = max_sent_len - filter_size + 1
            conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=filter_size)
            # TODO: Sequential
            component = nn.Sequential(
                conv1,
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=maxpool_kernel_size))

            conv_blocks.append(component)

        # TODO: ModuleList
        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.fc = nn.Linear(num_filters * len(filter_sizes), 64)
###############################################

    def forward_once(self, x):
        # x: (batch, sentence_len)
        x = self.word_embeddings(x)
        # x.shape: (batch, sent_len, embed_dim) --> (batch, embed_dim, sent_len)
        x = x.transpose(1, 2)  # switch 2nd and 3rd axis
        x_list = [conv_block(x) for conv_block in self.conv_blocks]

        # x_list.shape: [(num_filters, filter_size_3), (num_filters, filter_size_4), ...]
        out = torch.cat(x_list, 2)  # concatenate along filter_sizes
        out = out.view(out.size(0), -1)

        # feature_extracted = out
        out = F.dropout(out, p=self.dropout_prob, training=self.training)
        return self.fc(out)

    def forward(self, sen1, sen2):

        output1 = self.forward_once(sen1)
        output2 = self.forward_once(sen2)

        if(self.contra_loss):
            return output1, output2
        else:
            output = torch.cat((output1, output2),1)
            return output


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, eps=1e-8)
        loss_contrastive = torch.mean(0.2*(1 - label) * torch.pow(euclidean_distance, 2) +  # calmp夹断用法
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


def evaluate(data, model, loss):
    """
    Evaluation, return accuracy and loss
    """
    print("evaluate")
    model.eval()  # set mode to evaluation to disable dropout
    data_loader = DataLoader(data, batch_size=50)

    TP, FP, TN, FN = 0, 0, 0, 0
    pos_num, neg_num = 0, 0
    ii = 0
    total_loss = 0.0
    for data1, data2, label in data_loader:

        data1, data2, label = Variable(data1),Variable(data2), Variable(label)

        output1,output2 = model(data1,data2)
        #edis = F.pairwise_distance(output1, output2)
        #loss_contrastive = torch.mean((1 - label) * torch.pow(edis, 2) +  # calmp夹断用法
        #                               (label) * torch.pow(torch.clamp(2.0 - edis, min=0.0), 2))

        cosdis = F.cosine_similarity(output1, output2, dim = 1, eps = 1e-8)
        cc = 0.5

        for i in range(len(label)):
            if label[i] == 1: pos_num += 1
            else: neg_num += 1


        for i in range(len(label)):
            if cosdis[i] >= cc and label[i] == 1:  #
                TP += 1
            if cosdis[i] < cc and label[i] == 0:
                TN += 1
            if cosdis[i] >= cc and label[i] == 0:
                FP += 1
            if cosdis[i] < cc and label[i] == 1:
                FN += 1

        #total_loss += loss_contrastive

    PRECISION = float(TP) / float(TP + FP+1)
    RECALL = float(TP) / float(TP + FN+1)

    ACC = float(TP + TN) / float(TP + TN + FP + FN+1)
    F1 = float(2 * TP) / float(2 * TP + FN + FP+1)

    print neg_num, pos_num
    print TP,TN,FP,FN
    print "ACC,F1:",ACC,F1
    return total_loss,F1


def train():
    print "train"
    start_time = time.time()
    config = SiameseTCNNConfig()
    corpus = Corpus(train_file, vocab_file, 0.0, config.seq_length, config.vocab_size)
    testcorpus = Corpus(test_file, vocab_file, 1.0, config.seq_length, config.vocab_size)
    print(corpus)
    print(testcorpus)

    config.vocab_size = len(corpus.words)

    train_data = TensorDataset(torch.LongTensor(corpus.x_train1), torch.LongTensor(corpus.x_train2),
                               torch.FloatTensor(corpus.y_train))
    test_data = TensorDataset(torch.LongTensor(testcorpus.x_test1), torch.LongTensor(testcorpus.x_test2),
                              torch.FloatTensor(testcorpus.y_test))

    print('Configuring CNN model...')
    model = SiameseTextCNN(config)
    print(model)

    # optimizer and loss function
    # criterion = nn.CrossEntropyLoss(size_average=False)
    # criterion = torch.nn.BCELoss(reduce=False, size_average=False)
    if config.contra_loss:
        criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # set the mode to train
    print("Training and evaluating...")
    best_F1 = 0.0

    for epoch in range(config.num_epochs):
        # load the training data in batch
        model.train()
        train_loader = DataLoader(train_data, batch_size=config.batch_size)
        ii = 0
        for x1_batch, x2_batch, y_batch in train_loader:
            ii+=1
            if ii % 100 == 0: print epoch,"batch",ii
            inputs1, inputs2, targets = Variable(x1_batch), Variable(x2_batch), Variable(y_batch)

            optimizer.zero_grad()
            outputs1, outputs2 = model(inputs1, inputs2)  # forward computation

            loss = criterion(outputs1, outputs2, targets)
            """
            todo
            """
            # backward propagation and update parameters
            loss.backward()
            optimizer.step()

        # evaluate on both training and test dataset

        print "epoch",epoch
        train_loss, train_F1 = evaluate(train_data, model, criterion)
        test_loss, test_F1 = evaluate(test_data, model, criterion)
        #print "train_loss:",train_loss

        if test_F1 > best_F1:
             # store the best result
             best_F1 = test_F1
             improved_str = '*'
             torch.save(model.state_dict(), model_file)
        else:
             improved_str = ''

        time_dif = get_time_dif(start_time)
        msg = "Epoch {0:3}, Train_loss: {1:>7.3}, Train_F1 {2:>6.3%}, " \
              + "Test_loss: {3:>6.3}, Test_F1 {4:>6.3%}, Time: {5} {6}"
        print(msg.format(epoch + 1, train_loss, train_F1,
                         test_loss, test_F1, time_dif, improved_str))


def predict(sen1,sen2):
    # load config and vocabulary
    config = SiameseTCNNConfig()
    _, word_to_id = read_vocab(vocab_file)

    # load model
    model = SiameseTextCNN(config)
    model.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))

    # process text
    sen1 = process_text(sen1, word_to_id, config.seq_length)
    sen1 = Variable(torch.LongTensor([sen1]), volatile=True)

    sen2 = process_text(sen2, word_to_id, config.seq_length)
    sen2 = Variable(torch.LongTensor([sen2]), volatile=True)

    model.eval()  # very important
    output = model(sen1,sen2)

    output1, output2 = model(sen1, sen2)

    cosdis = F.cosine_similarity(output1, output2, dim=1, eps=1e-8)

    euclidean_distance = F.pairwise_distance(output1, output2, eps=1e-8)
    print cosdis
    #loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +  # calmp夹断用法
    #                              (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

    if(cosdis >= 0.5): return 1
    else: return 0

def submit():
    print "submit"
    TP, FP, TN, FN,Total = 0,0,0,0,0
    with open(test_file, 'r') as fin, open(result_file, 'w') as fout:
        for line in fin:
            lineno, sen1, sen2, tag = line.strip().split('\t')

            result = predict(sen1, sen2)
            # print lineno, sen1, sen2, tag, result
            print lineno,sen1,sen2,result,tag
            fout.write(lineno + '\t'+str(result)+'\n')
            Total += 1
            if result == 1 and tag == "1":  #
                TP += 1
            if result == 0 and tag == "0":
                TN += 1
            if result == 1 and tag == "0":
                FP += 1
            if result == 0 and tag == "1":
                FN += 1

        PRECISION = float(TP) / float(TP + FP)
        RECALL = float(TP) / float(TP + FN)

        ACC = float(TP + TN) / float(TP + TN + FP + FN)
        F1 = float(2 * TP) / float(2 * TP + FN + FP)
        F1X = 2 * PRECISION * RECALL / (PRECISION + RECALL)
        # print TP, FP, TN, FN

        print F1, ACC, PRECISION, RECALL

train()
submit()