import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertJapaneseTokenizer, BertModel
from sklearn.metrics import accuracy_score, mean_absolute_error, cohen_kappa_score
from argparse import ArgumentParser
from tqdm import tqdm


DEVICE_IDS = [0, 1]

RANDOM_SEED = 123

PATIENCE = 3

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

    def __getitem__(self, index):
        data_row = self.data.iloc[index]
        text = data_row[0]
        labels = data_row[1]

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
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=os.cpu_count())

    def valid_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=os.cpu_count())

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=os.cpu_count())


class EmotionClassifier(torch.nn.Module):
    def __init__(self, n_classes: int, drop_rate=DROP_RATE, pretrained_model=BERT_MODEL):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.drop = torch.nn.Dropout(drop_rate)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, n_classes)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        _, pooler_output = self.bert(input_ids, attention_mask=attention_mask)
        preds = self.classifier(self.drop(pooler_output))

        return preds


class EarlyStopping:
    def __init__(self, patience=PATIENCE, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.force_cancel = False

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...")
        torch.save(model.state_dict(), 'models/model.pth')
        self.val_loss_min = val_loss


def calculate_loss_and_scores(model, loader, criterion, device):
    model.eval()
    y_preds =[]
    y_true =[]
    loss = 0.0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            output = model(input_ids, attention_mask)
            loss += criterion(output, labels).item()
            y_preds.append(torch.argmax(output, dim=-1).cpu().numpy())
            y_true.append(labels.cpu().numpy())
            print(y_preds, y_true)

    return {
        'loss': loss / len(loader),
        'accuracy': accuracy_score(y_preds, y_true),
        'mean_absolute_error': mean_absolute_error(y_preds, y_true),
        'cohen_kappa_score': cohen_kappa_score(y_preds, y_true, weight='quadratic')
    }


def train_model(model, train_dataloader, valid_dataloader, criterion, optimizer, earlystopping, device, n_epochs=N_EPOCHS):
    train_log = []
    valid_log = []
    for epoch in range(n_epochs):
        model.train()
        for batch in tqdm(train_dataloader, desc=f"[Epoch {epoch + 1}]"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            output = model(input_ids, attention_mask)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
         
        train_scores = calculate_loss_and_scores(model, train_dataloader, criterion, device)
        valid_scores = calculate_loss_and_scores(model, valid_dataloader, criterion, device)
        train_log.append([train_scores['loss'], train_scores['accuracy_score']])
        valid_log.append([valid_scores['loss'], valid_scores['accuracy_score']])

        earlystopping(valid_scores['loss'], model)
        if earlystopping.early_stop:
            print("Early stopping")
            break
        
        print(f"train_loss: {train_scores['loss']:.3f}, train_accuracy: {train_scores['accuracy_score']:.3f}, valid_loss: {valid_scores['loss']:.3f}, valid_accuracy: {valid_scores['accuracy_score']:.3f}")

    return {
        'train_log': train_log,
        'valid_log': valid_log
    }


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--wrime', default='./data/pn-long.csv')
    parser.add_argument('--y_train_valid')
    parser.add_argument('--y_test')

    return parser.parse_args()


def torch_fix_seed(random_seed=RANDOM_SEED):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


def main():
    args = get_args()

    df = pd.read_csv(args.wrime, header=0)

    train_df = df[df['Train/Div/Test'] == 'train'].loc[:,['Sentence', args.y_train_valid]].reset_index(drop=True)
    valid_df = df[df['Train/Div/Test'] == 'dev'].loc[:,['Sentence', args.y_train_valid]].reset_index(drop=True)
    test_df = df[df['Train/Div/Test'] == 'test'].loc[:,['Sentence', args.y_test]].reset_index(drop=True)

    data_module = CreateDataModule(train_df, valid_df, test_df, batch_size=BATCH_SIZE, max_token_len=MAX_TOKEN_LEN, pretrained_model=BERT_MODEL)
    data_module.setup()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = EmotionClassifier(n_classes=N_CLASSES, drop_rate=DROP_RATE, pretrained_model=BERT_MODEL)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam([
        {'params': model.bert.parameters(), 'lr': LEARNING_RATE},
        {'params': model.classifier.parameters(), 'lr': LEARNING_RATE}
    ])

    earlystopping = EarlyStopping(
        patience=PATIENCE,
        verbose=True
    )

    train_model(model, data_module.train_dataloader(), data_module.valid_dataloader(), criterion, optimizer, earlystopping, device, n_epochs=N_EPOCHS)
    

if __name__ == "__main__":
    torch_fix_seed()
    main()


