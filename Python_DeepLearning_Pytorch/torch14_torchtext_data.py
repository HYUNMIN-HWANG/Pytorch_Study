# IMDb Dataset
# 간단한 Sentiment Analysis Task

from torchtext.legacy import data
from torchtext.legacy import datasets

# Data Setting
TEXT = data.Field(lower=True, batch_first=True)
# lower : 문장을 모두 소문자화하는 옵션
# batch_first : batch_size 를 가장 앞으로 설정하는 옵션
LABEL = data.Field(sequential=False)

train, test = datasets.IMDB.splits(TEXT, LABEL)
# splits : IMDB에 있는 데이터를 train, test 데이터셋으로 쉽게 가져올 수 있다.

print(train)    # <torchtext.legacy.datasets.imdb.IMDB object at 0x0000024E1482A828>
print(test)     # <torchtext.legacy.datasets.imdb.IMDB object at 0x0000024E1482AFD0>
