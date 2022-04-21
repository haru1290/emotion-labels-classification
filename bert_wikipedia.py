import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch import cuda
from torch.utils.data import Dataset, DataLoader
from transformers import BertJapaneseTokenizer, BertModel
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm


TEXT_COLUMN = 'Sentence'
LABEL_COLUMN = 'W_Joy'

DEVICE_IDS = [0, 1]

BERT_MODEL = 'cl-tohoku/bert-base-japanese-whole-word-masking'

N_EPOCHS = 3
N_CLASSES = 4
MAX_TOKEN_LEN = 128
BATCH_SIZE = 32
DROP_RATE = 0.1
LEARNING_RATE = 2e-5


class CreateDataset(Dataset):
    def __init__(self, data, tokenizer, max_token_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_row = self.data.iloc[idx]
        text = data_row[TEXT_COLUMN]
        labels = data_row[LABEL_COLUMN]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels)
        }


class CreateDataModule():
    def __init__(self, train_df, valid_df, test_df, batch_size=BATCH_SIZE, max_token_len=MAX_TOKEN_LEN, pretrained_model=BERT_MODEL):
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.max_token_len = max_token_len
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(pretrained_model)

    def setup(self):
        self.train_dataset = CreateDataset(self.train_df, self.tokenizer, self.max_token_len)
        self.valid_dataset = CreateDataset(self.valid_df, self.tokenizer, self.max_token_len)
        self.test_dataset = CreateDataset(self.test_df, self.tokenizer, self.max_token_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)

    def valid_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)


class EmotionClassifier(nn.Module):
    def __init__(self, n_classes: int, drop_rate=DROP_RATE, pretrained_model=BERT_MODEL):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.drop = torch.nn.Dropout(drop_rate)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, n_classes)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        _, output = self.bert(input_ids, attention_mask=attention_mask)
        preds = self.classifier(self.drop(output))

        return self.softmax(preds)


def calculate_loss_and_accuracy(model, loader, criterion, device):
    model.eval()
    loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            preds = model(input_ids, attention_mask)
            loss += criterion(preds, labels).item()

            preds = torch.argmax(preds, dim=-1).cpu().numpy()
            labels = labels.cpu().numpy()
            total += len(labels)
            correct += (preds == labels).sum().item()
    
    return loss / len(loader), correct / total


def train_model(model, train_dataloader, valid_dataloader, optimizer, criterion, device, n_epochs=N_EPOCHS):
    train_log = []
    valid_log = []
    for epoch in range(n_epochs):
        model.train()
        for batch in tqdm(train_dataloader):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            optimizer.zero_grad()
            preds = model(input_ids, attention_mask)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
         
        train_loss, train_accuracy = calculate_loss_and_accuracy(model, train_dataloader, criterion, device)
        valid_loss, valid_accuracy = calculate_loss_and_accuracy(model, valid_dataloader, criterion, device)
        train_log.append([train_loss, train_accuracy])
        valid_log.append([valid_loss, valid_accuracy])
        
        print(f"epoch: {epoch + 1}, train_loss: {train_loss:.3f}, train_accuracy: {train_accuracy:.3f}, valid_loss: {valid_loss:.3f}, valid_accuracy: {valid_accuracy:.3f}")

    return {
        'train_log': train_log,
        'valid_log': valid_log
    }


def main():
    df = pd.read_csv('./data/pn-long.csv', header=0)

    train_df = df[df['Train/Div/Test'] == 'train'].reset_index(drop=True)
    valid_df = df[df['Train/Div/Test'] == 'dev'].reset_index(drop=True)
    test_df = df[df['Train/Div/Test'] == 'test'].reset_index(drop=True)

    data_module = CreateDataModule(train_df, valid_df, test_df)
    data_module.setup()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = EmotionClassifier(n_classes=N_CLASSES, drop_rate=DROP_RATE, pretrained_model=BERT_MODEL)
    model = torch.nn.DataParallel(model, device_ids=DEVICE_IDS)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam([
        {'params': model.module.bert.parameters(), 'lr': LEARNING_RATE},
        {'params': model.module.classifier.parameters(), 'lr': LEARNING_RATE}
    ])

    train_model(model, data_module.train_dataloader(), data_module.valid_dataloader(), criterion, optimizer, device, n_epochs=N_EPOCHS)
    

if __name__ == "__main__":
    main()


