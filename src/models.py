import torch
import torch.nn as nn
from transformers import BertModel, BertForPreTraining


class BertWikiClassifier(nn.Module):
    def __init__(self, n_classes: int, drop_rate, pretrained_model, mode):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.drop = nn.Dropout(drop_rate)
        self.mode = mode
        if self.mode == 'Concat':
            self.classifier = nn.Linear(self.bert.config.hidden_size * 4, n_classes)
        else:
            self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask, user_features):
        _, pooler_output = self.bert(input_ids, attention_mask=attention_mask)
        if self.mode == 'Sum':
            sequence_output = pooler_output + user_features
        elif self.mode == 'Difference':
            sequence_output = pooler_output - user_features
        elif self.mode == 'Product':
            sequence_output = pooler_output * user_features
        elif self.mode == 'Concat':
            sequence_output_sum = pooler_output + user_features
            sequence_output_dif = pooler_output + user_features
            sequence_output_pro = pooler_output * user_features
            sequence_output = torch.cat([pooler_output, sequence_output_sum, sequence_output_dif, sequence_output_pro], dim=1)
        else:
            sequence_output = pooler_output
        preds = self.classifier(self.drop(sequence_output))

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
