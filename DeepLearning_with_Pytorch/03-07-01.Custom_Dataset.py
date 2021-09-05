# Custom Dataset
# torch.utils.data.Dataset 을 상속받아 직접 커스텀 데이터셋을 만드는 경우

import torch

class CustomDataset (torch.utils.data.Dataset):
    def __init__(self):
        # 데이터 전처리를 해주는 부분
    def __len__(self):
        # 데이터 셋의 길이, 총 샘플의 수를 적는 부분
    def __getitem__(self, idx):
        # 데이터 셋에서 특정 1개의 샘플을 가져오는 함수