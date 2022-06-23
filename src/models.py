#
# models.py
#
import torch.nn as nn
from transformers import BertModel


BERT_MODEL = 'cl-tohoku/bert-base-japanese-whole-word-masking'
DROP_RATE = 0.1


class BertClassifier(nn.Module):
    def __init__(self, n_classes: int, drop_rate=DROP_RATE, pretrained_model=BERT_MODEL):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.drop = nn.Dropout(drop_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooler_output = self.bert(input_ids, attention_mask=attention_mask)
        preds = self.classifier(self.drop(pooler_output))

        return preds

'''
class BertProductClassifier(torch.nn.Module):
    def __init__(self, n_classes: int, drop_rate=DROP_RATE, pretrained_model=BERT_MODEL):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.drop = torch.nn.Dropout(drop_rate)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooler_output = self.bert(input_ids, attention_mask=attention_mask)
        preds = self.classifier(self.drop(pooler_output))

        return preds


class BertConcatClassifier(torch.nn.Module):
    def __init__(self, n_classes: int, drop_rate=DROP_RATE, pretrained_model=BERT_MODEL):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.drop = torch.nn.Dropout(drop_rate)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooler_output = self.bert(input_ids, attention_mask=attention_mask)
        preds = self.classifier(self.drop(pooler_output))

        return preds
'''