# python3.5 buildtagger.py <train_file_absolute_path> <model_file_absolute_path>

import os
import math
import sys
import datetime
import numpy as np
import torch
from torch import optim
from torch import nn
from torch.utils import data
import torch.nn.functional as F
from collections import defaultdict
import re
import pickle
import random
import itertools
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

WORD_EMBEDDING_DIM = 200
CHAR_EMBEDDING_DIM = 50
CHAR_CONV_K = 3
CHAR_CONV_L = 100
CHAR_CONV_PADDING = 1
LSTM_HIDDEN_SIZE = 300
LSTM_NUM_LAYERS = 2
BATCH_SIZE = 64

# move to GPU if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: ', device)

def WT_list_preprocess(line_list):
    '''
    preprocess a line list
    '''
    words = []
    words_idx_list = []
    tags = []
    tags_idx_list = []
    for word_tag_pair in line_list:
        w = word_tag_pair[:word_tag_pair.rindex('/')]
        t = word_tag_pair[word_tag_pair.rindex('/')+1:]
        words.append(w)
        tags.append(t)
        if w in word_set:
            words_idx_list.append(word_to_idx_dict[w])
        else:
            words_idx_list.append(word_to_idx_dict['unknown'])
        tags_idx_list.append(tag_to_idx_dict[t])
    return words, tags, words_idx_list, tags_idx_list
        

class word_char_embding_model(nn.Module):
    def __init__(self, size_vocab, size_char):
        super(word_char_embding_model, self).__init__()

        self.word2vec = nn.Embedding(
            num_embeddings = size_vocab,
            embedding_dim = WORD_EMBEDDING_DIM,
            padding_idx = word_pad_value)
        self.char2vec = nn.Embedding(
            num_embeddings = size_char,
            embedding_dim = CHAR_EMBEDDING_DIM,
            padding_idx = char_pad_value)

        self.char_cnn = nn.Conv1d(in_channels = CHAR_EMBEDDING_DIM,
            out_channels = CHAR_CONV_L,
            kernel_size = CHAR_CONV_K,
            padding = CHAR_CONV_PADDING)

    def forward(self, padded_words_idx, padded_chars_idx, lengths):
        # char-level embding
        # [number of words, CHAR_EMBEDDING_DIM, max number of chars]
        c_embding = self.char2vec(padded_chars_idx).permute(0,2,1).contiguous()
        c_embding = self.char_cnn(c_embding)
        c_embding = c_embding.max(dim=-1)[0]
        w_embding = self.word2vec(padded_words_idx)
        add_c_embding = torch.zeros([BATCH_SIZE, w_embding.size(1),CHAR_CONV_L])
        start = 0
        for i_line, length in enumerate(lengths):
           add_c_embding[i_line, :length,:] = c_embding[start:start+length]
           start += length
        w_embding = torch.cat((w_embding, add_c_embding), dim=2)
        packed_w_embding = pack_padded_sequence(w_embding, lengths,\
            batch_first=True, enforce_sorted=False)
        return packed_w_embding



class POS_tag_model(nn.Module):
    def __init__(self):
        super(POS_tag_model, self).__init__()
        self.BiLSTM = nn.LSTM(input_size = WORD_EMBEDDING_DIM+CHAR_CONV_L,
                            hidden_size = LSTM_HIDDEN_SIZE,
                            num_layers = LSTM_NUM_LAYERS,
                            batch_first = True,
                            bidirectional = True)
        # self.h0 = torch.randn(LSTM_NUM_LAYERS * 2,
        #                     MAX_SENTENCE_SIZE,
        #                     LSTM_HIDDEN_SIZE)
        # self.c0 = torch.randn(LSTM_NUM_LAYERS * 2,
        #                     MAX_SENTENCE_SIZE,
        #                     LSTM_HIDDEN_SIZE)

        self.fc = nn.Linear(LSTM_HIDDEN_SIZE * 2, Num_tags)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x):
        x = x.view(1, x.size(0), x.size(1)).contiguous()
        x, (hn, cn) = self.BiLSTM(x)
        x = self.fc(x)
        tags = self.softmax(x)
        return tags


class joint_model(nn.Module):
    def __init__(self, size_vocab, size_char, size_tag):
        super(joint_model, self).__init__()

        # language model
        self.word2vec = nn.Embedding(
            num_embeddings = size_vocab,
            embedding_dim = WORD_EMBEDDING_DIM,
            padding_idx = word_pad_value)
        self.char2vec = nn.Embedding(
            num_embeddings = size_char,
            embedding_dim = CHAR_EMBEDDING_DIM,
            padding_idx = char_pad_value)

        self.char_cnn = nn.Conv1d(in_channels = CHAR_EMBEDDING_DIM,
            out_channels = CHAR_CONV_L,
            kernel_size = CHAR_CONV_K,
            padding = CHAR_CONV_PADDING)

        # POSTAG model
        self.BiLSTM = nn.LSTM(input_size = WORD_EMBEDDING_DIM+CHAR_CONV_L,
                            hidden_size = LSTM_HIDDEN_SIZE,
                            num_layers = LSTM_NUM_LAYERS,
                            batch_first = True,
                            bidirectional = True)
        self.fc = nn.Linear(LSTM_HIDDEN_SIZE * 2, Num_tags)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, padded_words_idx, padded_chars_idx, lengths):

        # language model
        # char-level embding
        # [number of words, CHAR_EMBEDDING_DIM, max number of chars]
        c_embding = self.char2vec(padded_chars_idx).permute(0,2,1).contiguous()
        c_embding = self.char_cnn(c_embding)
        c_embding = c_embding.max(dim=-1)[0]
        w_embding = self.word2vec(padded_words_idx)
        add_c_embding = torch.zeros([BATCH_SIZE, w_embding.size(1),CHAR_CONV_L])
        start = 0
        for i_line, length in enumerate(lengths):
           add_c_embding[i_line, :length,:] = c_embding[start:start+length]
           start += length
        w_embding = torch.cat((w_embding, add_c_embding), dim=2)
        packed_w_embding = pack_padded_sequence(w_embding, lengths,\
            batch_first=True, enforce_sorted=True)

        # POSTAG model
        packed_w_embding, (hn, cn) = self.BiLSTM(packed_w_embding)
        padded_w_embding, _ = pad_packed_sequence(packed_w_embding, batch_first=True) 
        # TODO this operation will sort the lines in decreasing oder

        #prediction
        tags = self.fc(padded_w_embding)
        tags = self.softmax(tags)
        return tags


def train_model(train_file, model_file):

    global char_set, word_set, tag_set
    global Num_words, Num_chars, Num_tags, Num_lines
    global char_to_idx_dict, word_to_idx_dict, tag_to_idx_dict
    global idx_to_char_dict, idx_to_word_dict, idx_to_tag_dict
    global word_pad_value, char_pad_value
    
    # # initialize sets
    # word_set = set() # word set
    # char_set = set() # char set
    # tag_set = set() # word tag set
    # training_words_list = []
    # training_chars_list = []
    # training_tags_list = []

    # # get char, word, tag set
    # with open(train_file) as f_in:
    #     lines = f_in.read()
    # lines_list = lines.split('\n')
    # Num_lines = len(lines_list)
    # for line in lines_list:
    #     line_list = line.split()
    #     word_list = [w_t_pair[:w_t_pair.rindex('/')] for w_t_pair in line_list]
    #     # char_list = [[c for c in w] for w in word_list]
    #     tag_list = [w_t_pair[w_t_pair.rindex('/')+1:] for w_t_pair in line_list]
    #     training_words_list.append(word_list)
    #     # training_chars_list.append(char_list)
    #     training_tags_list.append(tag_list)
    #     word_set.update(set(word_list))
    #     tag_set.update(set(tag_list))
    #     char_set.update(set([c for w in word_list for c in w]))

    # # indexing
    # char_to_idx_dict = {c: (i+1) for i, c in enumerate(char_set)}
    # idx_to_char_dict = {(i+1): c for i, c in enumerate(char_set)}
    # word_to_idx_dict = {w: (i+2) for i, w in enumerate(word_set)}
    # idx_to_word_dict = {(i+2): w for i, w in enumerate(word_set)}
    # tag_to_idx_dict = {t: i for i, t in enumerate(tag_set)}
    # idx_to_tag_dict = {i: t for i, t in enumerate(tag_set)}

    # #add unknown & pad
    # word_set.add('unkown')
    # word_set.add('<PAD>')
    # word_to_idx_dict['<PAD>'] = 0
    # idx_to_word_dict[0] = '<PAD>'
    # word_to_idx_dict['unknown'] = 1
    # idx_to_word_dict[1] = 'unknown'
    # char_set.add('<PAD>')
    # char_to_idx_dict['<PAD>'] = 0
    # idx_to_char_dict[0] = '<PAD>'

    # # calculate number of words, tags, chars, including unknown and <PAD>
    # Num_words = len(word_set) # 44391
    # Num_chars = len(char_set) # 85
    # Num_tags = len(tag_set) # 45

    # # all training examples convert to idxs
    # training_words_idx_list = [[word_to_idx_dict[w] for w in word_list]\
    #     for word_list in training_words_list]
    # training_chars_idx_list = [[[char_to_idx_dict[c] for c in w]\
    #     for w in word_list] for word_list in training_words_list]
    # training_tags_idx_list = [[tag_to_idx_dict[t] for t in tag_list]\
    #     for tag_list in training_tags_list]

    # idx_dicts = {}
    # idx_dicts['char_set'] = char_set
    # idx_dicts['word_set'] = word_set
    # idx_dicts['tag_set'] = tag_set
    # idx_dicts['char_to_idx_dict'] = char_to_idx_dict
    # idx_dicts['idx_to_char_dict'] = idx_to_char_dict
    # idx_dicts['word_to_idx_dict'] = word_to_idx_dict
    # idx_dicts['idx_to_word_dict'] = idx_to_word_dict
    # idx_dicts['tag_to_idx_dict'] = tag_to_idx_dict
    # idx_dicts['idx_to_tag_dict'] = idx_to_tag_dict
    # idx_dicts['training_words_list'] = training_words_list
    # idx_dicts['training_tags_list'] = training_tags_list
    # idx_dicts['training_words_idx_list'] = training_words_idx_list
    # idx_dicts['training_chars_idx_list'] = training_chars_idx_list
    # idx_dicts['training_tags_idx_list'] = training_tags_idx_list
    # with open('idx_dicts', 'wb') as f:
    #     pickle.dump(idx_dicts, f)

    # directly load following data for debugging efficiency
    idx_dicts = pickle.load(open("idx_dicts", "rb"))
    char_set = idx_dicts['char_set']
    word_set = idx_dicts['word_set']
    tag_set = idx_dicts['tag_set']
    char_to_idx_dict = idx_dicts['char_to_idx_dict']
    idx_to_char_dict = idx_dicts['idx_to_char_dict']
    word_to_idx_dict = idx_dicts['word_to_idx_dict']
    idx_to_word_dict = idx_dicts['idx_to_word_dict']
    tag_to_idx_dict = idx_dicts['tag_to_idx_dict']
    idx_to_tag_dict = idx_dicts['idx_to_tag_dict']
    training_words_list = idx_dicts['training_words_list']
    training_tags_list = idx_dicts['training_tags_list']
    training_words_idx_list = idx_dicts['training_words_idx_list']
    training_chars_idx_list = idx_dicts['training_chars_idx_list']
    training_tags_idx_list = idx_dicts['training_tags_idx_list']
    Num_words = len(word_set) # 44391
    Num_chars = len(char_set) # 85
    Num_tags = len(tag_set) # 45
    Num_lines = len(training_words_list) # 39833

    pad_token = '<PAD>'
    word_pad_value = word_to_idx_dict[pad_token]
    char_pad_value = char_to_idx_dict[pad_token]

    model = joint_model(size_vocab=Num_words, size_char=Num_chars, size_tag=Num_tags)
    adam = optim.Adam(model.parameters(), lr=0.001)
    loss = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')

    for i_epoch in range(5):
        Num_B = int(Num_lines/BATCH_SIZE)
        for i_batch in range(Num_B):
            # sort in decreasing order
            ori_order = [i for i in range(i_batch*BATCH_SIZE, (i_batch+1)*BATCH_SIZE)]
            sorted_order = sorted(ori_order, key = lambda i:len(training_words_idx_list[i]), reverse=True)

            # retrive a batch of words, tags, and chars
            batch_words_idx = [training_words_idx_list[i] for i in sorted_order]
            batch_chars_idx = [training_chars_idx_list[i] for i in sorted_order]
            batch_tags_idx = [training_tags_idx_list[i] for i in sorted_order]

            #pad them to the same length
            lengths = [len(s) for s in batch_words_idx]
            padded_words_idx = list(itertools.zip_longest(*batch_words_idx, fillvalue=word_pad_value))
            padded_words_idx = torch.LongTensor(padded_words_idx).permute(1,0)
            padded_tags_idx = list(itertools.zip_longest(*batch_tags_idx, fillvalue=-100))
            padded_tags_idx = torch.LongTensor(padded_tags_idx).permute(1,0)
            padded_chars_idx= [batch_chars_idx[i][j] for i in range(BATCH_SIZE) for j in range(lengths[i])]
            padded_chars_idx = list(itertools.zip_longest(*padded_chars_idx, fillvalue=char_pad_value))
            padded_chars_idx = torch.LongTensor(padded_chars_idx).permute(1,0)

            #train model
            pred_tags = model.forward(padded_words_idx,padded_chars_idx, lengths) # [1, number of words, 45]
            pred_tags = pred_tags.squeeze() # [number of words, 45]
            # tags_idx_onehot = F.one_hot(torch.LongTensor(tags_idx_list), Num_tags)
            # tags_idx_onehot = tags_idx_onehot.view(1,tags_idx_onehot.size(0),tags_idx_onehot.size(1)).contiguous()
            tags_idx = torch.LongTensor(tags_idx_list)
            loss_value = loss(pred_tags, tags_idx)

            print("Epoch:", i_epoch+1,
                  "Step:", i_line, '/', Num_lines,
                  "Loss:", loss_value.data.item())

            adam.zero_grad()
            loss_value.backward()
            adam.step()
        random,shuffle(lines_list)


    print('Finished...')

if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    start_time = datetime.datetime.now()
    train_model(train_file, model_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
