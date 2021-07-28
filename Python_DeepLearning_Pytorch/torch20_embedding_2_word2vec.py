# word2vec
"""
가정 : Token의 의미는 주변 Token의 정보로 표현된다.
문장을 윈도우 형태로 부분만 보는 것을 기본으로 시작
기준 Token의 양옆 Token을 포함한 윈도우가 이동하면서 윈도우 속 Token과 기준 Token의 관계를 학습시키는 과정을 진행함
Context(주변 Token) , Target(기준 Token)
    - CBoW : Context Token의 벡터로 변환해 더한 후 Target Token을 맞춘다.
    - Skip-Gram : Target Token을 벡터로 변환한 후 Context Token을 맞춘다.
장점 : 대부분 0이 아닌 값으로 채워진다, 벡터의 크기도 원-핫 인코딩의 크기보다 일반적으로 작은 값을 갖고 있다.
"""