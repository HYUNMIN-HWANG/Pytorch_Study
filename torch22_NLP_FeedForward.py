# RNN, LSTM, GRU

import re
import sys
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtext.legacy import data
from torchtext.legacy import datasets
from torchtext.legacy.data import dataset
from torchtext.legacy.data.iterator import batch

# Data Setting
# TEXT = data.Field(batch_first = True,
#                   fix_length = 500,
#                   tokenize=str.split,
#                   pad_first=True,
#                   pad_token='[PAD]',
#                   unk_token='[UNK]')

# LABEL = data.LabelField(dtype=torch.float)

TEXT = data.Field(sequential=True, 
                batch_first=True, 
                lower=True, 
                fix_length = 500, 
                tokenize=str.split, 
                pad_first=True,
                pad_token='<pad>',
                unk_token='<unk>')

LABEL = data.LabelField(dtype=torch.float)

train_data, test_data = datasets.IMDB.splits(text_field = TEXT, 
                                             label_field = LABEL)

# Data Length
print(f'Train Data Length : {len(train_data.examples)}')
print(f'Test Data Length : {len(test_data.examples)}')
# Train Data Length : 25000
# Test Data Length : 25000

print(train_data.fields)
# {'text': <torchtext.legacy.data.field.Field object at 0x000001AF35241F60>, 
# 'label': <torchtext.legacy.data.field.LabelField object at 0x000001AF3552E588>}

# Data Sample
print('---- Data Sample ----')
print('Input : ')
print(' '.join(vars(train_data.examples[1])['text']),'\n')
print('Label : ')
print(vars(train_data.examples[1])['label'])


# Pre-process Data
def PreProcessingText(input_sentence):
    input_sentence = input_sentence.lower() # 소문자화
    input_sentence = re.sub('<[^>]*>', repl= ' ', string = input_sentence) # "<br />" 처리
    input_sentence = re.sub('[!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~]', repl= ' ', string = input_sentence) # 특수문자 처리 ("'" 제외)
    input_sentence = re.sub('\s+', repl= ' ', string = input_sentence) # 연속된 띄어쓰기 처리
    if input_sentence:
        return input_sentence

for example in train_data.examples :
    vars(example)['text'] = PreProcessingText(' '.join(vars(example)['text'])).split()

for example in test_data.examples :
    vars(example)['text'] = PreProcessingText(' '.join(vars(example)['text'])).split()

# pre-trained
model_config = {'emb_type' : 'glove', 'emb_dim' : 300}

TEXT.build_vocab(train_data,
                 min_freq = 5, 
                 max_size = None,
                 vectors = f"glove.6B.{model_config['emb_dim']}d")

LABEL.build_vocab(train_data)

model_config['vocab_size'] = len(TEXT.vocab)


# Vocabulary Info
print(f'Vocab Size : {len(TEXT.vocab)}')

print('Vocab Examples : ')
for idx, (k, v) in enumerate(TEXT.vocab.stoi.items()):
    if idx >= 10:
        break    
    print('\t', k, v)

print('---------------------------------')

# Label Info
print(f'Label Size : {len(LABEL.vocab)}')

print('Lable Examples : ')
for idx, (k, v) in enumerate(LABEL.vocab.stoi.items()):
    print('\t', k, v)

print(TEXT.vocab.vectors.shape)
# torch.Size([51956, 300])

# Spliting Valid set
train_data, valid_data = train_data.split(random_state = random.seed(0),
                                          split_ratio=0.8)

model_config['batch_size'] = 30

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(datasets=(train_data, valid_data, test_data), 
                                                                           batch_size=model_config['batch_size'],shuffle=True, repeat=False,
                                                                           device=device)

print('훈련 데이터의 미니 배치의 개수 : {}'.format(len(train_iterator)))    # 667
print('테스트 데이터의 미니 배치의 개수 : {}'.format(len(test_iterator)))   # 834
print('검증 데이터의 미니 배치의 개수 : {}'.format(len(valid_iterator)))    # 167

# Checking feed-forward
sample_for_check = next(iter(train_iterator))
print(sample_for_check)
print(sample_for_check.text)
print(sample_for_check.label)

print("=====================================")

class SentenceClassification(nn.Module) :
    def __init__(self, **model_config):
        super(SentenceClassification, self).__init__()

        if model_config['emb_type'] == 'glove' or 'fasttext':
            self.emb = nn.Embedding(model_config['vocab_size'],
                                    model_config['emb_dim'],
                                    _weight = TEXT.vocab.vectors)
        else:
            self.emb = nn.Embedding(model_config['vocab_size'],
                                    model_config['emb_dim'])
        
        self.bidirectional = model_config['bidirectional']
        self.num_direction = 2 if model_config['bidirectional'] else 1
        self.model_type = model_config['model_type'] 

        self.RNN = nn.RNN (input_size = model_config['emb_dim'],
                           hidden_size = model_config['hidden_dim'],
                           dropout=model_config['dropout'],
                           bidirectional = model_config['bidirectional'],
                           batch_first = model_config['batch_first'])
        
        self.LSTM= nn.LSTM(input_size = model_config['emb_dim'],
                           hidden_size = model_config['hidden_dim'],
                           dropout=model_config['dropout'],
                           bidirectional = model_config['bidirectional'],
                           batch_first = model_config['batch_first'])
        
        self.GRU = nn.GRU (input_size = model_config['emb_dim'],
                           hidden_size = model_config['hidden_dim'],
                           dropout=model_config['dropout'],
                           bidirectional = model_config['bidirectional'],
                           batch_first = model_config['batch_first'])
    
        self.fc = nn.Linear(model_config['hidden_dim'] * self.num_direction,
                            model_config['output_dim'])
        
        self.drop = nn.Dropout(model_config['dropout'])
    
    def forward(self, x):
        
        emb = self.emb(x) 
        # emb : (Batch_Size, Max_Seq_Length, Emb_dim)

        if self.model_type == 'RNN':
            output, hidden = self.RNN(emb) 
        elif self.model_type == 'LSTM':
            output, (hidden, cell) = self.LSTM(emb)
        elif self.model_type == 'GRU':
            output, hidden = self.GRU(emb)
        else:
            raise NameError('Select model_type in [RNN, LSTM, GRU]')
        
        # output : (Batch_Size, Max_Seq_Length, Hidden_dim * num_direction) 
        # hidden : (num_direction, Batch_Size, Hidden_dim)
        
        last_output = output[:,-1,:]

        # last_output : (Batch_Size, Hidden_dim * num_direction)
        return self.fc(self.drop(last_output))


# Checking feed-forward
sample_for_check = next(iter(train_iterator))
print(sample_for_check)
print(sample_for_check.text)
print(sample_for_check.label)

"""
[torchtext.legacy.data.batch.Batch of size 30]
        [.text]:[torch.cuda.LongTensor of size 30x500 (GPU 0)]
        [.label]:[torch.cuda.FloatTensor of size 30 (GPU 0)]
tensor([[   1,    1,    1,  ...,    2,  812,   96],
        [   1,    1,    1,  ..., 1200,    5,   12],
        [   1,    1,    1,  ...,   53,   98, 2187],
        ...,
        [   1,    1,    1,  ...,   10,  368,   78],
        [   1,    1,    1,  ...,   19,    6,   64],
        [   1,    1,    1,  ..., 2571,    5,   99]], device='cuda:0')
tensor([1., 0., 1., 1., 0., 0., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0.,
        0., 0., 0., 0., 1., 1., 0., 0., 1., 1., 1., 1.], device='cuda:0')
"""

model_config.update(dict(batch_first = True,
                            model_type = 'RNN',     # 원하는 모델을 선택한다.
                            bidirectional = True,   # 양방향 모델을 사용할 것인지 설정
                            hidden_dim = 128,
                            output_dim = 1,         # binary classification이기 때문에 1로 설정함
                            dropout = 0.2))

model = SentenceClassification(**model_config).to(device)
loss_fn = nn.BCEWithLogitsLoss().to(device)

def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc

predictions = model.forward(sample_for_check.text).squeeze()
loss = loss_fn(predictions, sample_for_check.label)
acc = binary_accuracy(predictions, sample_for_check.label)

print("------------------------------")
print(predictions)
print("loss : ", loss)
print("acc : ", acc)

"""
tensor([ 0.0863,  0.1121, -0.0707, -0.1347,  0.0655, -0.0534, -0.0246,  0.0010,
        -0.0179,  0.0969, -0.1396,  0.1160,  0.0784,  0.2441,  0.1118, -0.0409,
        -0.2159,  0.1711,  0.1816,  0.0899,  0.1149, -0.0509,  0.0778, -0.1131,
         0.1206,  0.1668, -0.0414,  0.2070,  0.2651,  0.1489], device='cuda:0',
       grad_fn=<SqueezeBackward0>)
loss :  tensor(0.7085, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>)
acc :  tensor(0.4000, device='cuda:0')
"""

