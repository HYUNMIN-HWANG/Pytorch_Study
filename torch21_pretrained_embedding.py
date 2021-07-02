import re
import sys
import random

import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F

from torchtext.legacy import data
from torchtext.legacy import datasets
from torchtext.legacy.data.iterator import batch

# Data Setting
TEXT = data.Field(batch_first = True,       # batch_size 를 가장 앞으로 설정하는 옵션
                  fix_length = 500,         # sentence의 길이를 미리 제한하는 옵션
                  tokenize = str.split,     # 띄어쓰기 기반으로 tokenize
                  pad_first = True,         # 길이를 맞추기 위해서 padding을 앞에서 준다.
                  pad_token = '[PAD]',      # padding에 대한 특수 토큰
                  unk_token = '[UNK]')      # token dictionary에 없는 token이 나왔을 경우 해당 token을 표현하는 특수 토큰

LABEL = data.LabelField(dtype=torch.float)  # 

train_data, test_data = datasets.IMDB.splits(text_field = TEXT, label_field = LABEL)

# Data Length
print(f"Train Data Length : {len(train_data.examples)}")    # data.examples : 데이터의 개수를 확인할 수 있음
print(f"Test Data Length : {len(test_data.examples)}")
# Train Data Length : 25000
# Test Data Length : 25000

# Data Field
print(train_data.fields)
# {'text': <torchtext.legacy.data.field.Field object at 0x000001EB97BD1F60>, 
# 'label': <torchtext.legacy.data.field.LabelField object at 0x000001EB97E7AAC8>}

# Data Sample
print("=== Data Sample ===")
print("Input : ")
print(" ".join(vars(train_data.examples[1])['text']),'\n')
# vars :  객체의 어트리뷰트를 돌려준다 -> 데이터 값을 직접 확인할 수 있다.
print("Label : ")
print(vars(train_data.examples[1])['label'])
"""
=== Data Sample ===
Input :
Homelessness (or Houselessness as George Carlin stated) has been an issue for years but never a plan to help those on the street that were once considered human who did everything from going to school, work, or vote for the matter. Most people think of the homeless as just a lost cause while worrying about things such as racism, the war on Iraq, pressuring kids to succeed, technology, the elections, inflation, or worrying if they'll be next to end 
up on the streets.<br /><br />But what if you were given a bet to live on the streets for a month without the luxuries you once had from a home, the entertainment sets, a bathroom, pictures on the wall, a computer, and everything you once treasure to see what it's like to be homeless? That is Goddard Bolt's lesson.<br /><br />Mel Brooks (who directs) who stars as Bolt plays a rich man who has everything in the world until deciding to make a bet with a sissy rival (Jeffery Tambor) to see if he can live in the streets for thirty days without the luxuries; if Bolt succeeds, he can do what he wants with a future project of making more buildings. The bet's on where Bolt is thrown on the street with a bracelet on his leg to monitor his every move where he can't step off the sidewalk. He's given the nickname Pepto by a vagrant after it's written on his forehead where Bolt meets other characters including a woman by the name of Molly (Lesley Ann Warren) an ex-dancer who got divorce before losing her home, and her pals Sailor (Howard Morris) and Fumes (Teddy Wilson) who are already used to the streets. They're survivors. 
Bolt isn't. He's not used to reaching mutual agreements like he once did when being rich where it's fight or flight, kill or be killed.<br /><br />While the love connection between Molly and Bolt wasn't necessary to plot, I found "Life Stinks" to be one of Mel Brooks' observant films where prior to being a comedy, it shows a tender side compared to his slapstick work such as Blazing Saddles, Young Frankenstein, or Spaceballs for the matter, to show what it's like having something valuable before losing it the next day or on the other hand making a stupid bet like all rich people do when they don't know what to do with their money. Maybe they should give it to the homeless instead of using it like Monopoly money.<br /><br />Or maybe this film will inspire you to help others.

Label :
pos
"""

# Pre-process Data
def preprocessingText(input_sentence):
    input_sentence = input_sentence.lower() # 소문자화
    input_sentence = re.sub('<[^>]*>', repl=' ', string=input_sentence) # "<br />" 처리
    input_sentence = re.sub('[!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~]', repl= ' ', string = input_sentence)  # 특수문자처리
    input_sentence = re.sub('\s+', repl= ' ', string = input_sentence) # 연속된 띄어쓰기 처리
    if input_sentence :
        return input_sentence

for example in train_data.examples :
    vars(example)['text'] = preprocessingText(' '.join(vars(example)['text'])).split()

for example in test_data.examples :
    vars(example)['text'] = preprocessingText(' '.join(vars(example)['text'])).split()

# pre-trained
TEXT.build_vocab(train_data, 
                 min_freq=2,                # Vocab에 해당하는 Token에 최소한으로 등장하는 횟수에 제한을 둘 수 있음
                 max_size=None,             # 전체 vocab size 자체에도 제한을 둘 수 있음
                 vectors="glove.6B.300d")   # pre-trained vector를 가져와 vocab에 세팅하는 옵션

LABEL.build_vocab(train_data)

# Vocabulary Info
print(f'Vocab Size : {len(TEXT.vocab)}')

print("Voab Examples : ")
for idx, (k, v) in enumerate(TEXT.vocab.stoi.items()):
    if idx >= 10:
        break
    print('\t', k, v)

print("===================================")

# Label Info
print(f"Label Size : {len(LABEL.vocab)}")

print("Label Samples : ")
for idx, (k, v) in enumerate(LABEL.vocab.stoi.items()):
    if idx >= 10:
        break
    print('\t', k, v)
"""
Vocab Size : 51956
Voab Examples :
         [UNK] 0
         [PAD] 1
         the 2
         and 3
         a 4
         of 5
         to 6
         is 7
         in 8
         it 9
===================================
Label Size : 2
Label Samples :
         neg 0
         pos 1
"""

# check embedding vector
print(TEXT.vocab.vectors.shape)
# torch.Size([51956, 300])

# Spliting Valid set
train_data, valid_data = train_data.split(random_state=random.seed(0), split_ratio=0.8)

devic= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(datasets=(train_data, valid_data, test_data), 
                                                                            batch_size=30, 
                                                                            device=device)
                                                                            
                                                                            



                                                                            