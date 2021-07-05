# BERT 모델 데이터 적용하기
import re
import sys
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtext.legacy import data
from torchtext.legacy import datasets

from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

sentence = "My dog is cute. He likes playing. I bought a  pet food for him"
print(tokenizer.tokenize(sentence)) # ['my', 'dog', 'is', 'cute', '.', 'he', 'likes', 'playing', '.', 'i', 'bought', 'a', 'pet', 'food', 'for', 'him']
print(len(tokenizer.vocab)) # 30522

max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']
print(max_input_length) # 512

def new_tokenizer(sentence):
    tokens = tokenizer.tokenize(sentence) 
    tokens = tokens[:max_input_length-2]
    return tokens

def PreProcessingText(input_sentence):
    input_sentence = input_sentence.lower() # 소문자화
    input_sentence = re.sub('<[^>]*>', repl= ' ', string = input_sentence) # "<br />" 처리
    input_sentence = re.sub('[!"$%&\()*+,-./:;<=>?@[\\]^_`{|}~]', repl= ' ', string = input_sentence) # 특수문자 처리 ("'" 제외)
    input_sentence = re.sub('\s+', repl= ' ', string = input_sentence) # 연속된 띄어쓰기 처리
    if input_sentence:
        return input_sentence

def PreProc(list_sentence):
    return [tokenizer.convert_tokens_to_ids(PreProcessingText(x)) for x in list_sentence]


TEXT = data.Field(batch_first = True,
                  use_vocab = False,
                  tokenize = new_tokenizer,
                  preprocessing = PreProc,
                  init_token = tokenizer.cls_token_id,
                  eos_token = tokenizer.sep_token_id,
                  pad_token = tokenizer.pad_token_id,
                  unk_token = tokenizer.unk_token_id)

LABEL = data.LabelField(dtype = torch.float)

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
LABEL.build_vocab(train_data)
train_data, valid_data = train_data.split(random_state = random.seed(0), split_ratio=0.8)

'''Reading Data'''
# Data Length
print(f'Train Data Length : {len(train_data.examples)}')    # Train Data Length : 20000
print(f'Test Data Length : {len(test_data.examples)}')  # Test Data Length : 25000

# Data Fields
print(train_data.fields)
# {'text': <torchtext.legacy.data.field.Field object at 0x0000023011C71048>, 
# 'label': <torchtext.legacy.data.field.LabelField object at 0x0000023011C54DD8>}

# Data Sample
print('---- Data Sample ----')
print('Input : ')
print(tokenizer.convert_ids_to_tokens(vars(train_data.examples[2])['text']))
"""
Input :
['another', 'in', 'a', 'long', 'line', 'of', 'flick', '##s', 'made', 'by', 'people', 'who', 'think', 'that', 'knowing', 'how', 'to', 'operate', 'a', 'camera', 'is', 'the', 'same', 'as', 'telling', 'a', 'story', '[UNK]', 'within', '15', 'minutes', '[UNK]', 'the', 'entire', 'premise', 'is', 'laid', 'out', 'in', 'just', 'a', 'few', 'lines', '[UNK]', 'so', 'there', 'is', 'absolutely', 'no', 'mystery', '[UNK]', 'which', 'eliminate', '##s', 'a', 'whole', 'face', '##t', 'of', 'the', 'suspense', '[UNK]', 'the', 'only', 'half', '[UNK]', 'way', 'competent', 'actor', 'is', 'killed', '10', 'minutes', 'into', 'the', 'film', '[UNK]', 'so', 'we', "'", 're', 'left', 'with', 'stupid', 'characters', 'running', 'around', 'doing', 'stupid', 'things', '[UNK]', 'low', 'budget', 'films', 'can', "'", 't', 'afford', 'expensive', 'special', 'effects', '[UNK]', 'so', 'the', 'c', '##gi', 'portions', 'are', 'un', '##sur', '##pr', '##ising', '##ly', 'un', '##im', '##pressive', '[UNK]', 'but', 'were', 'at', 'least', 'a', 'valid', 'attempt', '[UNK]', 'the', 'creature', 'suit', 'is', 'terrible', '[UNK]', 'as', 'seen', 'when', 'it', 'falls', 'to', 'the', 'sidewalk', '[UNK]', 'and', 'the', 'director', 'keeps', 'emphasizing', 'the', 'eyes', '[UNK]', 'which', 'aren', "'", 't', 'even', 'the', 'red', 'color', 'shown', 'in', 'mirror', 'shots', '[UNK]', 'the', 'dialogue', 'is', 'clumsy', 'and', 'un', '##ins', '##pired', '[UNK]', 'with', 'some', 'lines', 'reminiscent', 'of', 'aliens', 'or', 'term', '##inator', '[UNK]', 'the', 'last', 'action', 'sequence', 'takes', 'place', 'in', 'a', 'police', 'station', '[UNK]', 'also', 'a', 'rip', '[UNK]', 'off', 'from', 'term', '##inator', '[UNK]', 'with', 'everyone', 'hiding', 'in', 'the', 'one', 'glass', 'lined', 'office', 'that', 'the', 'dark', '##wo', '##lf', 'doesn', "'", 't', 
'smash', 'into', '[UNK]', 'in', 'the', 'end', '[UNK]', 'the', 'girl', 'calls', 'the', 'hero', '[UNK]', 'a', 'good', 'protector', '[UNK]', '[UNK]', 'but', 'he', 'gets', 'both', 'his', 'partners', '[UNK]', 'the', 'original', 'protector', '[UNK]', 'and', 'at', 'least', 'three', 'other', 'civilians', '[UNK]', 'not', 'to', 'mention', 'a', 'dozen', 'cops', '[UNK]', 'all', 'killed', 'without', 'getting', 'a', 'decent', 'shot', 'off', '[UNK]', 'in', 'spite', 'of', 'an', 'arsenal', 'of', 'silver', 'bullets', 'and', 'a', 'sub', '##mac', '##hine', 'gun', '[UNK]', 'but', 'here', "'", 's', 'the', 'real', 'clinch', '##er', 'for', 'bad', 'writing', '[UNK]', 'they', 'could', 'have', 'killed', 'the', 'beast', 'right', 'after', 'the', 'beginning', 'credits', 'when', 'it', 'was', 'holding', 'the', 'strip', '##per', 'while', 'flashing', 'its', 'red', 'eyes', '[UNK]', 'instead', '[UNK]', 'they', 'took', 'it', 'into', 'custody', '[UNK]', '[UNK]', '[UNK]']
"""

'''Pre-processing Data'''

'''Making Vocab & Setting Embedding'''
# Label Info
print(f'Label Size : {len(LABEL.vocab)}')   # Label Size : 2

print('Lable Examples : ')
for idx, (k, v) in enumerate(LABEL.vocab.stoi.items()):
    print('\t', k, v)
# Lable Examples :
        #  neg 0
        #  pos 1

'''Spliting Validation Data & Making Data Iterator'''
model_config = {}
model_config['batch_size'] = 8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
                                                (train_data, valid_data, test_data), 
                                                batch_size=model_config['batch_size'],
                                                device=device)

# Check batch data
sample_for_check = next(iter(train_iterator))
print(sample_for_check)
print(sample_for_check.text)
print(sample_for_check.label)
# [torchtext.legacy.data.batch.Batch of size 8]
#         [.text]:[torch.cuda.LongTensor of size 8x512 (GPU 0)]
#         [.label]:[torch.cuda.FloatTensor of size 8 (GPU 0)]
# tensor([[ 101, 5752, 7104,  ...,    0,    0,    0],
#         [ 101,  100, 8334,  ...,    0,    0,    0],
#         [ 101, 1053,  100,  ...,    0,    0,    0],
#         ...,
#         [ 101, 1996, 2143,  ...,    0,    0,    0],
#         [ 101, 4931, 2065,  ...,    0,    0,    0],
#         [ 101, 2023, 3185,  ...,    0,    0,    0]], device='cuda:0')
# tensor([1., 1., 1., 0., 0., 0., 0., 1.], device='cuda:0')

'''Making Model'''
bert = BertModel.from_pretrained('bert-base-uncased')           # 원하는 BERT 모델을 가져와 사용할 수 있다.
model_config['emb_dim'] = bert.config.to_dict()['hidden_size']
print(model_config['emb_dim'])
# 768 >> Hidden layer size

class SentenceClassification(nn.Module):
    def __init__(self, **model_config):
        super(SentenceClassification, self).__init__()
        self.bert = bert
        self.fc = nn.Linear(model_config['emb_dim'], model_config['output_dim'])
    
    def forward(self, x):
        pooled_cls_output = self.bert(x)[1]
        return self.fc(pooled_cls_output)

'''Training'''
def train(model, iterator, optimizer, loss_fn, idx_epoch, **model_params):
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    batch_size = model_params['batch_size']

    for idx, batch in enumerate(iterator):
        # initializing
        optimizer.zero_grad()

        # forward
        predictions = model(batch.text).squeeze()
        loss = loss_fn(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)

        sys.stdout.write(
            "\r" + f"[Train] Epoch : {idx_epoch:^3}"\
                f"[{(idx+1) * batch_size} / {len(iterator) * batch_size} ({100. * (idx+1) / len(iterator) : .4}%)]"\
                    f" Loss : {loss.item():.4}"\
                        f" Acc : {acc.item():.4}"
        )

        # Backward
        loss.backward()
        optimizer.step()

        # Update Epoch Performance
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, loss_fn, idx_epoch, **model_params):
    epoch_loss = 0
    epoch_acc = 0

    batch_size = model.params['batch_size']

    # evaluate mode
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(iterator):
            predictions = model(batch.text).sqeeuze()
            loss = loss_fn(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

            sys.stdout.write(
                "\r" + f"[Eval] Epoch : {idx_epoch:^3}"\
                    f"[{(idx+1) * batch_size} / {len(iterator) * batch_size} ({100. * (idx+1) / len(iterator) :.4}%)]"\
                        f" Loss : {loss.item():.4}"\
                            f" Acc : {acc.item():.4}"
            )
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

'''bi-RNN'''
model_config.update(dict(output_dim=1))

def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum()/len(correct)
    return acc

model = SentenceClassification(**model_config)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
loss_fn = nn.BCEWithLogitsLoss().to(device)
model = model.to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(count_parameters(model))  # 109483009

N_EPOCH = 4

best_valid_loss = float('inf')
model_name = "BERT"

print("================================")
print(f"Model name : {model_name}")
print("================================")

for epoch in range(N_EPOCH) :
    train_loss, train_acc = train(model, train_iterator, optimizer, loss_fn, epoch, **model_config)
    print('')
    print(f"\t Epoch : {epoch} | Train Loss : {train_loss:.4} | Train Accuracy : {train_acc:.4}")
    valid_loss, valid_acc = evaluate(model, valid_iterator, loss_fn, epoch, **model_config)
    print('')
    print(f"\t Epoch : {epoch} | Valid Loss : {valid_loss:.4} | Valid Acc : {valid_acc:.4}")

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), f'../model_save/{model_name}.pt')
        print(f"\t Model is saved at {epoch}-epoch")

# Test set
epoch = 0
test_loss, test_acc = evaluate(model, test_iterator, loss_fn, epoch, **model_config)
print('')
print(f'Test Loss : {test_loss:.4} | Test Acc : {test_acc:.4}')











