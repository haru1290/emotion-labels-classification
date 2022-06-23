import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, mean_absolute_error, cohen_kappa_score
from argparse import ArgumentParser
from tqdm import tqdm

from make_dataset import CreateDataModule
from models import BertClassifier


class EarlyStopping:
    def __init__(self, best_model, patience, verbose=False):
        self.best_model = best_model
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_qwk_max = np.Inf
        self.force_cancel = False

    def __call__(self, val_qwk, model):
        score = val_qwk

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_qwk, model)
        elif score < self.best_score:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_qwk, model)
            self.counter = 0

    def save_checkpoint(self, val_qwk, model):
        if self.verbose:
            print(f"Validation qwk increased ({self.val_qwk_max:.3f} --> {val_qwk:.3f}). Saving model ...")

        torch.save(model.state_dict(), self.best_model)
        self.val_qwk_max = val_qwk


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
            y_preds += torch.argmax(output, dim=-1).cpu().tolist()
            y_true += labels.cpu().tolist()

    return {
        'loss': loss / len(loader),
        'accuracy_score': accuracy_score(y_preds, y_true),
        'mean_absolute_error': mean_absolute_error(y_preds, y_true),
        'cohen_kappa_score': cohen_kappa_score(y_preds, y_true, weights='quadratic')
    }


def train_step(model, train_dataloader, valid_dataloader, criterion, optimizer, earlystopping, n_epochs, device):
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
        train_log.append(train_scores['cohen_kappa_score'])
        valid_log.append(valid_scores['cohen_kappa_score'])

        print(f"train_qwk: {train_scores['cohen_kappa_score']:.3f},  valid_qwk: {valid_scores['cohen_kappa_score']:.3f}")

        earlystopping(valid_scores['cohen_kappa_score'], model)
        if earlystopping.early_stop:
            print("Early stopping")
            break

    return {
        'train_log': train_log,
        'valid_log': valid_log
    }


def main(args):
    train = pd.read_csv(args.train, header=0, sep='\t')
    valid = pd.read_csv(args.valid, header=0, sep='\t')
    test = pd.read_csv(args.test, header=0, sep='\t')

    data_module = CreateDataModule(
        train, valid, test,
        batch_size=args.batch_size,
        max_token_len=args.max_token_len,
        pretrained_model=args.pretrained
    )
    data_module.setup()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = BertClassifier(
        n_classes=args.n_class,
        drop_rate=args.drop_rate,
        pretrained_model=args.pretrained
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam([
        {'params': model.bert.parameters(), 'lr': args.learning_rate},
        {'params': model.classifier.parameters(), 'lr': args.learning_rate}
    ])
    
    earlystopping = EarlyStopping(
        best_model=args.best_model,
        patience=args.patience,
        verbose=args.verbose
    )

    train_step(model, data_module.train_dataloader(), data_module.valid_dataloader(), criterion, optimizer, earlystopping, args.n_epochs, device)

    model.load_state_dict(torch.load(args.best_model))
    test_scores = calculate_loss_and_scores(model, data_module.test_dataloader(), criterion, device)
    print(f"[Test] ACC: {test_scores['accuracy_score']:.3f}, MAE: {test_scores['mean_absolute_error']:.3f}, QWK: {test_scores['cohen_kappa_score']:.3f}")


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--train', default='./data/train.tsv')
    parser.add_argument('--valid', default='./data/valid.tsv')
    parser.add_argument('--test', default='./data/test.tsv')

    parser.add_argument('--pretrained', default='cl-tohoku/bert-base-japanese-whole-word-masking')

    parser.add_argument('--n_epochs', default=100)
    parser.add_argument('--n_class', default=5)
    parser.add_argument('--max_token_len', default=128)
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--drop_rate', default=0.1)
    parser.add_argument('--learning_rate', default=2e-5)

    parser.add_argument('--patience', default=3)
    parser.add_argument('--verbose', default=True)
    parser.add_argument('--best_model', default='./models/best_model.pth')

    args = parser.parse_args()

    main(args)
