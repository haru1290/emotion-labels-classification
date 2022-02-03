import os, sys
import itertools
import numpy as np
import pandas as pd
import transformers
import tensorflow as tf
import torch
import time
from torch.utils.data import Dataset, DataLoader
from transformers import BertJapaneseTokenizer, BertModel
from torch import optim
from torch import cuda
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import cohen_kappa_score


MAX_LEN = 128

DROP_RATE = 0.1
OUTPUT_SIZE = 5
BATCH_SIZE = 32
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5


class CreateDataset(Dataset):
    def __init__(self, X, y, tokenizer, max_len):
        self.X = X
        self.y = y
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        text = self.X[index]
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.LongTensor(ids),
            'mask': torch.LongTensor(mask),
            'labels': torch.LongTensor(self.y[index])
        }


class BERTClass(torch.nn.Module):
    def __init__(self, pretrained, drop_rate, otuput_size):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained)
        self.drop = torch.nn.Dropout(drop_rate)
        self.fc = torch.nn.Linear(768, otuput_size)

    def forward(self, ids, mask):
        _, out = self.bert(ids, attention_mask=mask)
        out = self.fc(self.drop(out))

        return out


def calculate_loss_and_accuracy(model, loader, device, criterion=None):
    model.eval()
    loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for data in loader:
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            labels = torch.flatten(data['labels'].to(device))

            outputs = model(ids, mask)
            
            if criterion != None:
                loss += criterion(outputs, labels).item()

            pred = torch.argmax(outputs, dim=-1).cpu().numpy()
            labels = labels.cpu().numpy()
            total += len(labels)
            correct += (pred == labels).sum().item()
    
    return loss / len(loader), correct / total


def train_model(dataset_train, dataset_dev, batch_size, model, criterion, optimizer, num_epochs, device=None):
    model.to(device)

    dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_dev = DataLoader(dataset=dataset_dev, batch_size=batch_size, shuffle=False)

    log_train = []
    log_dev = []
    for epoch in range(num_epochs):
        s_time = time.time()

        model.train()
        for data in dataloader_train:
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            labels = torch.flatten(data['labels'].to(device))

            optimizer.zero_grad()

            outputs = model(ids, mask)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        loss_train, acc_train = calculate_loss_and_accuracy(model, dataloader_train, device, criterion=criterion)
        loss_dev, acc_dev = calculate_loss_and_accuracy(model, dataloader_dev, device, criterion=criterion)
        log_train.append([loss_train, acc_train])
        log_dev.append([loss_dev, acc_dev])

        e_time = time.time()

        print(f'epoch: {epoch + 1}, loss_train: {loss_train:.4f}, accuracy_train: {acc_train:.4f}, loss_dev: {loss_dev:.4f}, accuracy_dev: {acc_dev:.4f}, {(e_time - s_time):.4f}sec')
    
    torch.save(model.state_dict(), "models/tohoku-BERT_W_PN.pth")

    return {'train': log_train, 'dev': log_dev}


def preprocessing(num):
    return int(num) + 2


def main():
    args = sys.argv

    df = pd.read_csv('data/PN_Splitdata.csv', header=0)

    emotion1 = args[1]
    emotion2 = args[2]

    df[emotion1] = df[emotion1].map(lambda x: preprocessing(x))
    if emotion1 != emotion2:
        df[emotion2] = df[emotion2].map(lambda x: preprocessing(x))
    
    train = df.loc[df['Train/Div/Test'].isin(['train']),['Sentence', emotion1]]
    dev = df.loc[df['Train/Div/Test'].isin(['dev']),['Sentence', emotion1]]
    test = df.loc[df['Train/Div/Test'].isin(['test']),['Sentence', emotion2]]

    train.reset_index(drop=True, inplace=True)
    dev.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    y_train = [[v] for v in list(train[emotion1])]
    y_dev = [[v] for v in list(dev[emotion1])]
    y_test = [[v] for v in list(test[emotion2])]

    pretrained = 'cl-tohoku/bert-base-japanese-whole-word-masking'

    tokenizer = BertJapaneseTokenizer.from_pretrained(pretrained)

    dataset_train = CreateDataset(train['Sentence'], y_train, tokenizer, MAX_LEN)
    dataset_dev = CreateDataset(dev['Sentence'], y_dev, tokenizer, MAX_LEN)
    dataset_test = CreateDataset(test['Sentence'], y_test, tokenizer, MAX_LEN)

    model = BERTClass(pretrained, DROP_RATE, OUTPUT_SIZE)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    device = 'cuda' if cuda.is_available() else 'cpu'

    log = train_model(dataset_train, dataset_dev, BATCH_SIZE, model, criterion, optimizer, NUM_EPOCHS, device=device)
    
    # テスト開始
    model.load_state_dict(torch.load("models/tohoku-BERT_W_PN.pth"))
    model.eval()

    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

    y_pred = []
    y_true = []
    for data in dataloader_test:
        ids = data['ids'].to(device)
        mask = data['mask'].to(device)
        labels = torch.flatten(data['labels'].to(device))

        outputs = model(ids, mask)
        
        y_pred.append(torch.argmax(outputs, dim=-1).cpu().numpy())
        y_true.append(labels.cpu().numpy())

    print(emotion1)
    print(emotion2)
    print('MAE:', mean_absolute_error(y_true, y_pred))
    print('QWK:', cohen_kappa_score(y_true, y_pred, weights='quadratic'))
    print('ACC:', accuracy_score(y_true, y_pred))
    

if __name__ == "__main__":
    main()


