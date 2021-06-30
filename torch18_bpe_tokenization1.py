# BPE Byte Pair Encoding
# 반복적으로 나오는 데이터의 연속된 패턴을 치환하는 방식을 사용해 데이터를 좀 더 효율적으로 저장하는 개념


# Algorithm 1 : Learn BPE operations
import re, collections
def get_stats(vocab):
    paris = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            paris[symbols[i],symbols[i+1]] += freq
    return paris

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\\S)' + bigram + r'(?!\\S)')
    for word in v_in :
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

vocab = {"l o w <\w>" : 5, "l o w e r <\w>" : 2, "n e w e s t <\w>" : 6, "w i d e s t <\w>" : 3}

num_merges = 10
for i in range(num_merges):
    print(f'Step {i+1}')
    pairs = get_stats(vocab)
    print("<pairs> ", pairs)
    best = max(pairs, key=pairs.get)
    print("<best> ", best)
    vocab = merge_vocab(best, vocab)
    print("<vocab> ", vocab)
    print("\n")


"""
Step 1
<pairs>  defaultdict(<class 'int'>, {('l', 'o'): 7, ('o', 'w'): 7, ('w', '<\\w>'): 5, ('w', 'e'): 8, ('e', 'r'): 2, ('r', '<\\w>'): 2, ('n', 'e'): 6, ('e', 'w'): 6, ('e', 's'): 9, ('s', 
't'): 9, ('t', '<\\w>'): 9, ('w', 'i'): 3, ('i', 'd'): 3, ('d', 'e'): 3})
<best>  ('e', 's')
<vocab>  {'l o w <\\w>': 5, 'l o w e r <\\w>': 2, 'n e w es t <\\w>': 6, 'w i d es t <\\w>': 3}


Step 2
<pairs>  defaultdict(<class 'int'>, {('l', 'o'): 7, ('o', 'w'): 7, ('w', '<\\w>'): 5, ('w', 'e'): 2, ('e', 'r'): 2, ('r', '<\\w>'): 2, ('n', 'e'): 6, ('e', 'w'): 6, ('w', 'es'): 6, ('es', 't'): 9, ('t', '<\\w>'): 9, ('w', 'i'): 3, ('i', 'd'): 3, ('d', 'es'): 3})
<best>  ('es', 't')
<vocab>  {'l o w <\\w>': 5, 'l o w e r <\\w>': 2, 'n e w est <\\w>': 6, 'w i d est <\\w>': 3}


Step 3
<pairs>  defaultdict(<class 'int'>, {('l', 'o'): 7, ('o', 'w'): 7, ('w', '<\\w>'): 5, ('w', 'e'): 2, ('e', 'r'): 2, ('r', '<\\w>'): 2, ('n', 'e'): 6, ('e', 'w'): 6, ('w', 'est'): 6, ('est', '<\\w>'): 9, ('w', 'i'): 3, ('i', 'd'): 3, ('d', 'est'): 3})
<best>  ('est', '<\\w>')
<vocab>  {'l o w <\\w>': 5, 'l o w e r <\\w>': 2, 'n e w est<\\w>': 6, 'w i d est<\\w>': 3}


Step 4
<pairs>  defaultdict(<class 'int'>, {('l', 'o'): 7, ('o', 'w'): 7, ('w', '<\\w>'): 5, ('w', 'e'): 2, ('e', 'r'): 2, ('r', '<\\w>'): 2, ('n', 'e'): 6, ('e', 'w'): 6, ('w', 'est<\\w>'): 6, ('w', 'i'): 3, ('i', 'd'): 3, ('d', 'est<\\w>'): 3})
<best>  ('l', 'o')
<vocab>  {'lo w <\\w>': 5, 'lo w e r <\\w>': 2, 'n e w est<\\w>': 6, 'w i d est<\\w>': 3}


Step 5
<pairs>  defaultdict(<class 'int'>, {('lo', 'w'): 7, ('w', '<\\w>'): 5, ('w', 'e'): 2, ('e', 'r'): 2, ('r', '<\\w>'): 2, ('n', 'e'): 6, ('e', 'w'): 6, ('w', 'est<\\w>'): 6, ('w', 'i'): 3, ('i', 'd'): 3, ('d', 'est<\\w>'): 3})
<best>  ('lo', 'w')
<vocab>  {'low <\\w>': 5, 'low e r <\\w>': 2, 'n e w est<\\w>': 6, 'w i d est<\\w>': 3}


Step 6
<pairs>  defaultdict(<class 'int'>, {('low', '<\\w>'): 5, ('low', 'e'): 2, ('e', 'r'): 2, ('r', '<\\w>'): 2, ('n', 'e'): 6, ('e', 'w'): 6, ('w', 'est<\\w>'): 6, ('w', 'i'): 3, ('i', 'd'): 3, ('d', 'est<\\w>'): 3})
<best>  ('n', 'e')
<vocab>  {'low <\\w>': 5, 'low e r <\\w>': 2, 'ne w est<\\w>': 6, 'w i d est<\\w>': 3}


Step 7
<pairs>  defaultdict(<class 'int'>, {('low', '<\\w>'): 5, ('low', 'e'): 2, ('e', 'r'): 2, ('r', '<\\w>'): 2, ('ne', 'w'): 6, ('w', 'est<\\w>'): 6, ('w', 'i'): 3, ('i', 'd'): 3, ('d', 'est<\\w>'): 3})
<best>  ('ne', 'w')
<vocab>  {'low <\\w>': 5, 'low e r <\\w>': 2, 'new est<\\w>': 6, 'w i d est<\\w>': 3}


Step 8
<pairs>  defaultdict(<class 'int'>, {('low', '<\\w>'): 5, ('low', 'e'): 2, ('e', 'r'): 2, ('r', '<\\w>'): 2, ('new', 'est<\\w>'): 6, ('w', 'i'): 3, ('i', 'd'): 3, ('d', 'est<\\w>'): 3}) 
<best>  ('new', 'est<\\w>')
<vocab>  {'low <\\w>': 5, 'low e r <\\w>': 2, 'newest<\\w>': 6, 'w i d est<\\w>': 3}


Step 9
<pairs>  defaultdict(<class 'int'>, {('low', '<\\w>'): 5, ('low', 'e'): 2, ('e', 'r'): 2, ('r', '<\\w>'): 2, ('w', 'i'): 3, ('i', 'd'): 3, ('d', 'est<\\w>'): 3})
<best>  ('low', '<\\w>')
<vocab>  {'low<\\w>': 5, 'low e r <\\w>': 2, 'newest<\\w>': 6, 'w i d est<\\w>': 3}


Step 10
<pairs>  defaultdict(<class 'int'>, {('low', 'e'): 2, ('e', 'r'): 2, ('r', '<\\w>'): 2, ('w', 'i'): 3, ('i', 'd'): 3, ('d', 'est<\\w>'): 3})
<best>  ('w', 'i')
<vocab>  {'low<\\w>': 5, 'low e r <\\w>': 2, 'newest<\\w>': 6, 'wi d est<\\w>': 3}
"""