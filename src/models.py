#
# models.py
#
import torch
from transformers import BertModel


BERT_MODEL = 'cl-tohoku/bert-base-japanese-whole-word-masking'
DROP_RATE = 0.1


class BertClassifier(torch.nn.Module):
    def __init__(self, n_classes: int, drop_rate=DROP_RATE, pretrained_model=BERT_MODEL):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.drop = torch.nn.Dropout(drop_rate)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, n_classes)
        # self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, input_ids, attention_mask):
        _, pooler_output = self.bert(input_ids, attention_mask=attention_mask)
        preds = self.classifier(self.drop(pooler_output))

        # return self.softmax(preds)
        return preds
