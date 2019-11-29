# python3.5 runtagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

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
import itertools
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

WORD_EMBEDDING_DIM = 300
CHAR_EMBEDDING_DIM = 60
CHAR_CONV_K = 3
CHAR_CONV_L = 300
CHAR_CONV_PADDING = 1
LSTM_HIDDEN_SIZE = 1024
LSTM_NUM_LAYERS = 2
DROPOUT_RATE = 0.5
BATCH_SIZE = 64

# move to GPU if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: ', device)

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
        add_c_embding = torch.zeros([w_embding.size(0), w_embding.size(1),CHAR_CONV_L]).to(device)
        start = 0
        for i_line, length in enumerate(lengths):
           add_c_embding[i_line, :length,:] = c_embding[start:start+length]
           start += length
        w_embding = torch.cat((w_embding, add_c_embding), dim=2)
        packed_w_embding = pack_padded_sequence(w_embding, lengths,\
            batch_first=True)

        # POSTAG model
        packed_w_embding, (hn, cn) = self.BiLSTM(packed_w_embding)
        padded_w_embding, _ = pad_packed_sequence(packed_w_embding, batch_first=True) 
        # TODO this operation will sort the lines in decreasing oder

        #prediction
        tags = F.dropout(padded_w_embding, DROPOUT_RATE)
        tags = self.fc(padded_w_embding)
        tags = self.softmax(tags)
        return tags

def tag_sentence(test_file, model_file, out_file):
    # write your code here. You can add functions as well.
    
    global Num_words, Num_chars, Num_tags
    global word_pad_value, char_pad_value

    # load data
    idx_dicts, model_state_dict = torch.load(model_file)
    # char_set = idx_dicts['char_set']
    word_set = idx_dicts['word_set']
    # tag_set = idx_dicts['tag_set']
    char_to_idx_dict = idx_dicts['char_to_idx_dict']
    # idx_to_char_dict = idx_dicts['idx_to_char_dict']
    word_to_idx_dict = idx_dicts['word_to_idx_dict']
    # idx_to_word_dict = idx_dicts['idx_to_word_dict']
    # tag_to_idx_dict = idx_dicts['tag_to_idx_dict']
    idx_to_tag_dict = idx_dicts['idx_to_tag_dict']
    
    pad_token = '<PAD>'
    word_pad_value = word_to_idx_dict[pad_token]
    char_pad_value = char_to_idx_dict[pad_token]
    Num_words = len(word_set)
    Num_chars = len(char_to_idx_dict)
    Num_tags = len(idx_to_tag_dict)
    
    # load test examples
    # get char, word, tag set
    test_lines_list = []
    with open(test_file) as f_in:
        lines = f_in.read()
    lines_list = lines.split('\n')
    # remove empty set
    while '' in lines_list:
        lines_list.remove('')
    Num_lines = len(lines_list) # 1993
    for line in lines_list:
        line_list = line.split()
        test_lines_list.append(line_list)

    # words and chars idx list
    test_words_idx_list = [[word_to_idx_dict[w] if w in word_set\
        else word_to_idx_dict['unknown']\
        for w in word_list] for word_list in test_lines_list]
    test_chars_idx_list = [[[char_to_idx_dict[c] for c in w]\
        for w in word_list] for word_list in test_lines_list]
    
    # instantiate model
    model = joint_model(size_vocab=Num_words, size_char=Num_chars, size_tag=Num_tags).to(device)
    model.load_state_dict(model_state_dict)
    
    # predict in batches
    test_lengths = [len(sent) for sent in test_words_idx_list]
    Num_B = int(np.ceil(Num_lines/BATCH_SIZE))
    for i_batch in range(Num_B):
        if i_batch == (Num_B-1):
            ori_order = list(range(i_batch*BATCH_SIZE, Num_lines))
        else:
            ori_order = list(range(i_batch*BATCH_SIZE, (i_batch+1)*BATCH_SIZE))
        sorted_order = sorted(ori_order, key = lambda i:test_lengths[i], reverse=True)
        
        # retrive a batch of words, tags, and chars
        batch_words_idx = [test_words_idx_list[i] for i in sorted_order]
        batch_chars_idx = [test_chars_idx_list[i] for i in sorted_order]
        
        #pad them to the same length
        lengths = [len(s) for s in batch_words_idx]
        padded_words_idx = list(itertools.zip_longest(*batch_words_idx, fillvalue=word_pad_value))
        padded_words_idx = torch.LongTensor(padded_words_idx).permute(1,0).to(device)
        padded_chars_idx= [batch_chars_idx[i][j] for i in range(len(lengths)) for j in range(lengths[i])]
        padded_chars_idx = list(itertools.zip_longest(*padded_chars_idx, fillvalue=char_pad_value))
        padded_chars_idx = torch.LongTensor(padded_chars_idx).permute(1,0).to(device)

        # predicate
        pred_tags = model.forward(padded_words_idx,padded_chars_idx, lengths)
        pred_tags = pred_tags.max(dim=-1)[1].to('cpu')
        
        # print
        converted_tags_idx = [pred_tags[sorted_order.index(i),:] for i in ori_order]
        output = ''
        for i in ori_order:
            try:
                single_output = ' '.join([test_lines_list[i][j]+'/'+idx_to_tag_dict[int(converted_tags_idx[i-i_batch*BATCH_SIZE][j])]\
                               for j in range(len(test_lines_list[i]))])
                output = output + single_output + '\n'
            except:
                ipdb.set_trace()
        with open(out_file,'a') as f_out:
            f_out.write(output)

    print('Finished...')

if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    start_time = datetime.datetime.now()
    tag_sentence(test_file, model_file, out_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)