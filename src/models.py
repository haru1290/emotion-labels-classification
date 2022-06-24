import torch
import torch.nn as nn
from transformers import BertModel, BertForPreTraining


class BertWikiClassifier(nn.Module):
    def __init__(self, n_classes: int, drop_rate, pretrained_model, mode):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.drop = nn.Dropout(drop_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.mode = mode

    def forward(self, input_ids, attention_mask, user_features):
        _, pooler_output = self.bert(input_ids, attention_mask=attention_mask)

        if self.mode == 'Product':
            pooler_output = pooler_output * user_features
        elif self.mode == 'Concat':
            pooler_output = torch.unsqueeze([pooler_output, user_features], dim=1)

        preds = self.classifier(self.drop(pooler_output))

        return preds


class BertSNSClassifier(nn.Module):
    def __init__(self, n_classes: int, drop_rate, pretrained_model):
        super().__init__()
        self.bert = BertForPreTraining.from_pretrained(pretrained_model)
        self.drop = nn.Dropout(drop_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask, user_features):
        _, pooler_output = self.bert(input_ids, attention_mask=attention_mask)
        preds = self.classifier(self.drop(pooler_output))

        return preds
