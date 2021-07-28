# BERT 모델

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

