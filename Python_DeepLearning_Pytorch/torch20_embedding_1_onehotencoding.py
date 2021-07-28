S1 = "나는 책상 위에 사과를 먹었다"
S2 = "알고 보니 그 사과는 Jason 것이었다"
S3 = "그래서 Jason에게 사과를 했다"

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

# 모든 token을 원-핫 인코딩으로 표현
# [1] python list를 이용

V = len(token2idx)
print(V)    # 14

token2vec = [([0 if i != idx else 1 for i in range(V)],idx,token) for token, idx in token2idx.items()]
# idx 에 해당하는 리스트 위치만 1로 넣는다.
for x in token2vec :
    print("\t".join([str(y) for y in x]))

"""
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]      0       나는
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]      1       책상
[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]      2       위에
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]      3       사과를
[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]      4       먹었다
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]      5       알고
[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]      6       보니
[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]      7       그
[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]      8       사과는
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]      9       Jason
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]      10      것이었다
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]      11      그래서
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]      12      Jason에게
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]      13      했다
"""

# [2] python numpy를 이용
import numpy as np
for sentence in [S1, S2, S3] :
    onehot_s = []
    tokens = sentence.split()
    for token in tokens :
        if token2idx.get(token) != None:
            vector = np.zeros((1,V))    # [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
            vector[:,token2idx[token]] = 1
            onehot_s.append(vector)
        else :
            print("UNK")
    
    print(f"{sentence} : ")
    print(np.concatenate(onehot_s, axis=0))
    print('\n')

"""
나는 책상 위에 사과를 먹었다 : 
[[1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]


알고 보니 그 사과는 Jason 것이었다 :
[[0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]]


그래서 Jason에게 사과를 했다 :
[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
 [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]
"""