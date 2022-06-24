import torch
import torch.nn as nn
from transformers import BertModel


class BertWikiClassifier(nn.Module):
    def __init__(self, n_classes: int, drop_rate, pretrained_model, mode):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.drop = nn.Dropout(drop_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.mode = mode

    def forward(self, input_ids, attention_mask, user_features):
        _, pooler_output = self.bert(input_ids, attention_mask=attention_mask)

        '''tmp_lst = torch.empty(len(pooler_outputs), 1536).to('cuda:0')
        if self.mode == 'Product':
            for i, pooler_output in enumerate(pooler_outputs):
                tmp_lst[i] = pooler_output * self.user_features[user_ids[i]]
        elif self.mode == 'Concat':
            for i, pooler_output in enumerate(pooler_outputs):
                tmp_lst[i] = torch.cat((pooler_output, self.user_features[user_ids[i]]), 0)
        preds = self.classifier(self.drop(tmp_lst))'''
        preds = self.classifier(self.drop(pooler_output))

        return preds
