# 글자 character를 token으로 사용한다.

# 영어 unicode
print([chr(k) for k in range(65, 91)])  # 영어 대문자
# ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
print([chr(k) for k in range(97, 123)]) # 영어 소문자
# ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

# 특수 문자 및 숫자 unicode
print([chr(k) for k in range(32, 48)]) 
# [' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/']
print([chr(k) for k in range(58, 65)]) 
# [':', ';', '<', '=', '>', '?', '@']
print([chr(k) for k in range(91, 97)]) 
# ['[', '\\', ']', '^', '_', '`']
print([chr(k) for k in range(123, 127)])
# ['{', '|', '}', '~'] 
print([chr(k) for k in range(48, 58)]) 
# ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# 한국어 unicode
print([chr(k) for k in range(int('0xAC00', 16), int('0xD7A3',16)+1)]) # 모든 완성형 (11,172자)
# ['가', '각', '갂', '갃', '간', '갅', '갆', '갇', '갈', '갉', '갊', '갋', '갌', '갍', '갎', '갏', '감', '갑', '값', '갓', '갔', '강', '갖', '갗', '갘', '같', '갚', '갛', '개', '객', '갞', '갟', '갠', '갡', '갢', '갣', '갤', '갥', '갦', ...
print([chr(k) for k in range(int('0x3131', 16), int('0x3163',16)+1)]) # 자음 모음
# ['ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ', 'ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']


# Character Token Vocabulary
idx2char = {0:'<pad>',1:'<unk>'}

str_idx = len(idx2char)
for x in range(32, 127) :
    idx2char.update({str_idx:chr(x)})
    str_idx += 1

# 한글 추가
for x in range(int('0x3131', 16), int('0x3163',16)+1):
    idx2char.update({str_idx:chr(x)})
    str_idx += 1

for x in range(int('0xAC00', 16), int('0xD7A3',16)+1):
    idx2char.update({str_idx:chr(x)})
    str_idx += 1

char2idx = {v:k for k, v in idx2char.items()}
print([char2idx.get(c,0) for c in "그래서 Jason에게 사과를 했다."])
# [652, 3116, 5552, 2, 44, 67, 85, 81, 80, 6756, 288, 2, 5440, 400, 3600, 2, 10780, 1912, 16]
print([char2idx.get(c,0) for c in "ㅇㅋ! ㄱㅅㄱㅅ"])
# [119, 123, 3, 2, 97, 117, 97, 117]

# 단점 : 표현법에 대한 학습이 어렵다.
# 글자보다 의미를 가진 단위이지 기존의 띄어쓰기보다 효과적인 단위를 찾아내는 방법을 연구해야 함
