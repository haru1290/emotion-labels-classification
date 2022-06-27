import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertJapaneseTokenizer


class CreateDataset(Dataset):
    def __init__(self, data, tokenizer, max_token_len, user_features):
        self.data = data
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.user_features = user_features

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_row = self.data.iloc[index]
        text = data_row['Sentence']
        labels = data_row['Avg. Readers_Sentiment']
        user_features = self.user_features[data_row['UserID'] - 1]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels),
            'user_features': user_features,
        }


class CreateDataModule():
    def __init__(self, train, valid, test, batch_size, max_token_len, pretrained_model, user_features):
        self.train = train
        self.valid = valid
        self.test = test
        self.batch_size = batch_size
        self.max_token_len = max_token_len
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(pretrained_model)
        self.user_features = user_features

    def setup(self):
        self.train_dataset = CreateDataset(self.train, self.tokenizer, self.max_token_len, self.user_features)
        self.valid_dataset = CreateDataset(self.valid, self.tokenizer, self.max_token_len, self.user_features)
        self.test_dataset = CreateDataset(self.test, self.tokenizer, self.max_token_len, self.user_features)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def valid_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=len(self.valid_dataset), shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=len(self.test_dataset), shuffle=False)
