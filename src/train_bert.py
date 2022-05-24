#
# train_bert.py
#
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, mean_absolute_error, cohen_kappa_score
from argparse import ArgumentParser
from tqdm import tqdm

from make_dataset import CreateDataModule
from models import BertClassifier


RANDOM_SEED = 123

PATIENCE = 3
BEST_MODEL_PATH = './models/best_model.pth'

BERT_MODEL = 'cl-tohoku/bert-base-japanese-whole-word-masking'

N_EPOCHS = 15
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
        self.val_qwk_max = np.Inf
        self.force_cancel = False

    def __call__(self, val_qwk, model):
        # score = -val_qwk
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

        torch.save(model.state_dict(), BEST_MODEL_PATH)
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


def train_step(model, train_dataloader, valid_dataloader, criterion, optimizer, earlystopping, device, n_epochs=N_EPOCHS):
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
        batch_size=BATCH_SIZE,
        max_token_len=MAX_TOKEN_LEN,
        pretrained_model=BERT_MODEL
    )
    data_module.setup()

    data_module.train_dataloader()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

    train_step(model, data_module.train_dataloader(), data_module.valid_dataloader(), criterion, optimizer, earlystopping, device, n_epochs=N_EPOCHS)

    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    test_scores = calculate_loss_and_scores(model, data_module.test_dataloader(), criterion, device)
    print(f"[Test] ACC: {test_scores['accuracy_score']:.3f}, MAE: {test_scores['mean_absolute_error']:.3f}, QWK: {test_scores['cohen_kappa_score']:.3f}")


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--train', default='./data/train.tsv')
    parser.add_argument('--valid', default='./data/valid.tsv')
    parser.add_argument('--test', default='./data/test.tsv')

    args = parser.parse_args()

    main(args)
