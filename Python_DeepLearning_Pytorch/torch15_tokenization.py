# 주어진 문자를 숫자로 표현해야 한다.
# 1. 문장을 의미 있는 부분으로 나눈다.
# 2. 나눠진 의미 있는 부분을 숫자로 바꿔 문장을 숫자로 표현한다.

S1 = "나는 책상 위에 사과를 먹었다"
S2 = "알고 보니 그 사과는 Jason 것이었다"
S3 = "그래서 Jason에게 사과를 했다"

# [1]
# 띄어쓰기 기준으로 문장 나누기 "Tokenization"
# 연속된 문자의 나열을 적절하게 의미를 지닌 부분의 나열로 바꾸는 과정
print(S1.split())   # ['나는', '책상', '위에', '사과를', '먹었다']
print(S2.split())   # ['알고', '보니', '그', '사과는', 'Jason', '것이었다']
print(S3.split())   # ['그래서', 'Jason에게', '사과를', '했다']

# Token 에 index를 지정해 사전 형식으로 모아본다.
token2idx = {}
index = 0

for sentence in [S1, S2, S3] :
    tokens = sentence.split()
    for token in tokens :
        if token2idx.get(token) == None :
            token2idx[token] = index
            index += 1

print(token2idx)
# {'나는': 0, '책상': 1, '위에': 2, '사과를': 3, '먹었다': 4, 
# '알고': 5, '보니': 6, '그': 7, '사과는': 8, 'Jason': 9, 
# '것이었다': 10, '그래서': 11, 'Jason에게': 12, '했다': 13}
# -> 문자를 숫자로 바꾸는 데 사용된다.

# [2]
# Token의 해당 숫자로 각 문장을 바꿔본다.
def indexed_sentence(sentence) :
    return [token2idx[token] for token in sentence]

S1_i = indexed_sentence(S1.split())
print(S1_i)
# [0, 1, 2, 3, 4]

S2_i = indexed_sentence(S2.split())
print(S2_i)
# [5, 6, 7, 8, 9, 10]

S3_i = indexed_sentence(S3.split())
print(S3_i)
# [11, 12, 3, 13]

# [3]
# Corpus & out-of-Vocabulary

# 새로운 문장, 사전에 없는 단어가 나오면 어떻게 처리하는가?
# out-of-Vocabulary : Token에 저장해둔 vocabulary에 token이 없어서 처음 본 token 이 나오는 현상
S4 = "나는 책상 위에 배를 먹었다"
# indexed_sentence(S4.split())
# KeyError: '배를'

# vacabulary에 없는 token 이 나올 경우 <unk>로 변환하도록 처리
token2idx = {t:i+1 for t, i in token2idx.items()}
token2idx['<unk>'] = 0

# token이 없을 경우 <unk> token의 0을 치환
def indexed_sentence_unk(sentence):
    return [token2idx.get(token, token2idx['<unk>']) for token in sentence]

S4_i = indexed_sentence_unk(S4.split())
print(S4_i) 
# [1, 2, 3, 0, 5]