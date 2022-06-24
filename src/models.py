import torch
import torch.nn as nn
from transformers import BertModel


class BertWikiClassifier(nn.Module):
    def __init__(self, n_classes: int, drop_rate, pretrained_model):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.drop = nn.Dropout(drop_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask, user_features):
        _, pooler_output = self.bert(input_ids, attention_mask=attention_mask)
        # pooler_output = pooler_output * user_features
        preds = self.classifier(self.drop(pooler_output))

        return preds
        