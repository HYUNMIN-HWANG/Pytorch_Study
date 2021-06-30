# Token Vocabulary를 이용해 문장을 어떻게 Tokenization 하는지 살펴보자

# import sentencepiece as spm
# s = spm.SentencePieceProcessor(model_file='spm.model')
# for n in range(5):
#     s.encode('New York', out_type=str, enable_sampling=True, alpha=0.1, nbest=-1)


# BERT에서 사용할 Tokenizer 사용법을 알아보겠다.
from threading import Semaphore
from transformers import BertTokenizer

sentence = "My dog is cute. He likes playing"

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # bert-base-uncased 라는 이름의 이미 학습된 모델을 사용할 것임
print(len(tokenizer.vocab)) # Tokenizer의 크기를 확인
# 30522
print(tokenizer.tokenize(sentence))
# ['my', 'dog', 'is', 'cute', '.', 'he', 'likes', 'playing'] ->split과 거의 다르지 않은 결과를 보임

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased') # 다양한 언어를 담고 있는 학습된 모델
print(len(tokenizer.vocab))
# 105879
print(tokenizer.tokenize(sentence))
# ['my', 'dog', 'is', 'cut', '##e', '.', 'he', 'likes', 'playing']
# ##e : 바로 이어지는 토큰을 의미함

sentence = "나는 책상 위에 사과를 먹었다. 알고 보니 그 사과는 Jason 것이었다. 그래서 Jason에게 사과를 했다."
print(tokenizer.tokenize(sentence))
# ['나는', 'ᄎ', '##ᅢᆨ', '##상', '위에', 'ᄉ', '##ᅡ', '##과', '##를', 'ᄆ', '##ᅥ', '##ᆨ', '##었다', '.', '알', '##고', 'ᄇ', '##ᅩ', '##니', '그', 'ᄉ', 
# '##ᅡ', '##과', '##는', 'jason', '것이', '##었다', '.', '그', '##래', '##서', 'jason', '##에게', 'ᄉ', '##ᅡ', '##과', '##를', '했다', '.']
# 다양한 언어들을 통틀어 최적의 Tokne을 구해야 하기 때문에, Token이 많이 잘리는 듯한 결과를 보여준다.
