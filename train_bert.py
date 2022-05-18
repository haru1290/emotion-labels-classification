#
# train_bert.py
#
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertJapaneseTokenizer, BertModel
from sklearn.metrics import accuracy_score, mean_absolute_error, cohen_kappa_score
from argparse import ArgumentParser
from tqdm import tqdm

from make_dataset import CreateDataModule
from models import BertClassifier


RANDOM_SEED = 123

PATIENCE = 3
MODEL_PATH = 'models/model.pth'

BERT_MODEL = 'cl-tohoku/bert-base-japanese-whole-word-masking'

N_EPOCHS = 10
N_CLASSES = 4
MAX_TOKEN_LEN = 128
BATCH_SIZE = 32
DROP_RATE = 0.1
LEARNING_RATE = 2e-5


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
            print(f"Validation loss decreased ({self.val_loss_min:.3f} --> {val_loss:.3f}). Saving model ...")
            
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
        for batch in tqdm(train_dataloader, desc=f"[Train epoch {epoch + 1}]"):
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


def main(args):
    train = pd.read_csv(args.train, header=0, sep='\t')
    valid = pd.read_csv(args.valid, header=0, sep='\t')
    test = pd.read_csv(args.test, header=0, sep='\t')

    data_module = CreateDataModule(train, valid, test, batch_size=BATCH_SIZE, max_token_len=MAX_TOKEN_LEN, pretrained_model=BERT_MODEL)
    data_module.setup()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = BertClassifier(
        n_classes=N_CLASSES,
        drop_rate=DROP_RATE,
        pretrained_model=BERT_MODEL
    )
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
    

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--train')
    parser.add_argument('--valid')
    parser.add_argument('--test')

    args = parser.parse_args()

    main(args)
