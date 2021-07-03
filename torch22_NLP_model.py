# RNN, LSTM, GRU

import re
import sys
import random
from unicodedata import bidirectional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtext.legacy import data
from torchtext.legacy import datasets
from torchtext.legacy.data import dataset
from torchtext.legacy.data.iterator import batch

# Data Setting
TEXT = data.Field(
    batch_first=True,
    fix_length=500,
    tokenize=str.split,
    pad_first=True,
    pad_token='[PAD]',
    unk_token='[UNK]'
)

LABEL = data.LabelField(dtype=torch.float)

train_data, test_data = datasets.IMDB.splits(text_field=TEXT, label_field=LABEL)

# Pre-process Data
def PreprocessingText(input_sentence):
    input_sentence = input_sentence.lower()
    input_sentence = re.sub('<[^>]*>', repl=' ', string=input_sentence)
    input_sentence = re.sub('[!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~]', repl= ' ', string = input_sentence) 
    input_sentence = re.sub('\s+', repl= ' ', string = input_sentence) 
    if input_sentence :
        return input_sentence

for example in train_data.examples :
    vars(example)['text'] = PreprocessingText(' '.join(vars(example)['text'])).split()

for example in test_data.examples :
    vars(example)['text'] = PreprocessingText(' '.join(vars(example)['text'])).split()

# pre-trained
TEXT.build_vocab(train_data,
                min_freq=2,
                max_siz=None,
                vectors="glove.6B.300d")

LABEL.build_vocab(train_data)

# Spliting Valid set
train_data, valid_data = train_data.split(random_state=random.seed(0), split_ratio=0.8)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    datasets=(train_data, valid_data, test_data),
    batch_size=30,
    device=device
)
    

class SentenceClassification(nn.Module) :
    def __init__(self, **model_config):
        super(SentenceClassification, self).__init__()

        if model_config['emb_type'] == 'glove' or 'fasttext' :
            self.emb = nn.Embedding(model_config['vocab_size'],
                                    model_config['emb_dim'],
                                    _weight = TEXT.vacab.vectors)
        else :
            self.emb = nn.Embedding(model_config['vocab_size'],
                                    model_config['emb_dim'])

        self.bidirectional = model_config['bidirectional']
        self.num_directrion = 2 if model_config['bidirectional'] else 1
        self.model_type = model_config['model_type']

        self.RNN = nn.RNN(input_size = model_config['emb_dim'],
                            hidden_size = model_config['hidden_dim'],
                            dropout = model_config['dropout'],
                            bidirectional = model_config['bidirectional'],
                            batch_first = model_config['batch_first'])

        self.LSTM = nn.LSTM(input_size = model_config['emb_dim'],
                            hidden_size = model_config['hidden_dim'],
                            dropout = model_config['dropout'],
                            bidirectional = model_config['bidirectional'],
                            batch_first = model_config['batch_first'])
            
        self.GRU = nn.GRU(input_size = model_config['emb_dim'],
                            hidden_size = model_config['hidden_dim'],
                            dropout = model_config['dropout'],
                            bidirectional = model_config['bidirectional'],
                            batch_first = model_config['batch_first'])
        
        self.fc = nn.Linear(model_config['hidden_dim']*self.num_directrion, model_config['output_dim'])

        self.drop = nn.Dropout(model_config['dropout'])
    
    def forward(self, x) :
        # x : (Batch_size, Max_Seq_Length)

        emb = self.emb(x)
        # emb : (Batch_size, Max_Seq_Length, Emb_dim)

        if self.model_type == 'RNN':
            output, hidden = self.RNN(emb)
        elif self.model_type == 'LSTM' :
            output, hidden = self.LSTM(emb)
        elif self.model_type == 'GRU':
            output, hidden = self.GRU(emb)
        else :
            raise NameError('Select model_type in [RNN, LSTM, GRU]')
        
        # output : (Batch_size, Max_Seq_Length, Hidden_dim * num_direction)
        # hidden : (num_direction, Batch_size, Hidden_dim)

        last_output = output[:,-1,:]

        # last_output : (Batch_size, Hidden_dim * num_direction)
        return self.fc(self.drop(last_output))

        


        