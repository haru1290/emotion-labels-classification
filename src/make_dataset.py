import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertJapaneseTokenizer


class CreateDataset(Dataset):
    def __init__(self, data, tokenizer, max_token_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_row = self.data.iloc[index]
        text = data_row['Sentence']
        labels = data_row['Writer_Sentiment']
        user_ids = data_row['UserID']

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels),
            'user_ids': torch.tensor(user_ids)
        }


class CreateDataModule():
    def __init__(self, train, valid, test, batch_size, max_token_len, pretrained_model):
        self.train = train
        self.valid = valid
        self.test = test
        self.batch_size = batch_size
        self.max_token_len = max_token_len
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(pretrained_model)

    def setup(self):
        self.train_dataset = CreateDataset(self.train, self.tokenizer, self.max_token_len)
        self.valid_dataset = CreateDataset(self.valid, self.tokenizer, self.max_token_len)
        self.test_dataset = CreateDataset(self.test, self.tokenizer, self.max_token_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=os.cpu_count())

    def valid_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=len(self.valid_dataset), shuffle=False, num_workers=os.cpu_count())

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=len(self.test_dataset), shuffle=False, num_workers=os.cpu_count())
