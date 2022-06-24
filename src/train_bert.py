import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, mean_absolute_error, cohen_kappa_score
from argparse import ArgumentParser
from tqdm import tqdm

from make_dataset import CreateDataModule
from models import BertWikiClassifier
from earlystopping import EarlyStopping


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
            user_features = batch['user_features'].to(device)

            output = model(input_ids, attention_mask, user_features)
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
    for epoch in range(n_epochs):
        model.train()
        for batch in tqdm(train_dataloader, desc=f"[Epoch {epoch + 1}]"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            user_features = batch['user_features'].to(device)

            optimizer.zero_grad()
            output = model(input_ids, attention_mask, user_features)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        
        valid_scores = calculate_loss_and_scores(model, valid_dataloader, criterion, device)
        earlystopping(valid_scores['cohen_kappa_score'], model)
        if earlystopping.early_stop:
            print("Early stopping")
            break


def torch_fix_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


def main(args):
    train = pd.read_csv(args.train, header=0, sep='\t')
    valid = pd.read_csv(args.valid, header=0, sep='\t')
    test = pd.read_csv(args.test, header=0, sep='\t')

    data_module = CreateDataModule(
        train, valid, test,
        batch_size=args.batch_size,
        max_token_len=args.max_token_len,
        pretrained_model=args.pretrained,
        user_features=torch.load('data/user_features.pt', map_location=torch.device('cpu'))
    )
    data_module.setup()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = BertWikiClassifier(
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

    torch_fix_seed()
    main(args)
