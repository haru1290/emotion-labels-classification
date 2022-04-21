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


# MODE = "product"

BERT_MODEL = 'cl-tohoku/bert-base-japanese-whole-word-masking'

N_EPOCHS = 3
N_CLASSES = 4
MAX_TOKEN_LEN = 128
BATCH_SIZE = 32
DROP_RATE = 0.1
LEARNING_RATE = 2e-5


class CreateDataset(Dataset):
    def __init__(self, X, y, tokenizer, max_token_len):
        # self.uids = uids
        self.X = X
        self.y = y
        self.tokenizer = tokenizer
        self.max_len = max_token_len

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # uids = self.uids[idx]
        text = self.X[idx]
        labels = self.y[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return dict(
            text=text,
            input_ids=encoding['input_ids'].flatten(),
            attention_mask=encoding['attention_mask'].flatten(),
            labels=torch.tensor(labels)
        )


class CreateDataModule():
    def __init__(self, n_classes: int, drop_rate=None, pretrained_model=BERT_MODEL):
        path


class BERTClass(nn.Module):
    def __init__(self, n_classes: int, drop_rate=None, pretrained_model=BERT_MODEL):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.drop = torch.nn.Dropout(drop_rate)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, output = self.bert(input_ids, attention_mask=attention_mask)
        print(_, output)
        preds = self.classifier(self.drop(output))

        return preds
        
        '''
        lst = torch.empty(len(out), 768).to(device)
        if mode == 'product':
            for i in range(len(out)):
                lst[i] = out[i] * mlp_feature[uids[i][0].item()-1]
            out = lst
        elif mode == 'concat':
            for i in range(len(out)):
                lst[i] = torch.cat((out[i], mlp_feature[uids[i][0].item()-1]), 0)
            out = lst
        '''


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


# def train_model(model, dataset_train, dataset_valid, criterion, batch_size=BATCH_SIZE, optimizer, num_epochs, model_path, mlp_feature, device=None):
def train_model(model, dataset_train, dataset_valid, criterion, optimizer, n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, device=None):
    model.to(device)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)

    log_train = []
    log_valid = []
    for epoch in range(n_epochs):
        model.train()
        for batch in dataloader_train:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            preds = model(input_ids, attention_mask, device)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
         
        loss_train, acc_train = calculate_loss_and_accuracy(model, dataloader_train,)
        loss_valid, acc_dev = calculate_loss_and_accuracy(model, dataloader_valid,)
        log_train.append([loss_train, acc_train])
        log_valid.append([loss_valid, acc_dev])
        
        print(f'epoch: {epoch + 1}, loss_train: {loss_train:.4f}, accuracy_train: {acc_train:.4f}, loss_dev: {loss_dev:.4f}, accuracy_dev: {acc_dev:.4f}, {(e_time - s_time):.4f}sec')
    
    # torch.save(model.state_dict(), model_path)

    return {'train': log_train, 'valid': log_valid}


def main():
    args = sys.argv

    df = pd.read_csv('data/pn-long.csv', header=0)

    emotion1 = args[1]
    #model_path = args[2]

    train = df.loc[df['Train/Div/Test'].isin(['train']),['UserID', 'Sentence', emotion1]]
    dev = df.loc[df['Train/Div/Test'].isin(['dev']),['UserID', 'Sentence', emotion1]]
    test = df.loc[df['Train/Div/Test'].isin(['test']),['UserID', 'Sentence', emotion1]]

    '''
    test_Joy = df.loc[df['Train/Div/Test'].isin(['test']),['UserID', 'Sentence', 'Avg.R_Joy']]
    test_Sadness = df.loc[df['Train/Div/Test'].isin(['test']),['UserID', 'Sentence', 'Avg.R_Sadness']]
    test_Anticipation = df.loc[df['Train/Div/Test'].isin(['test']),['UserID', 'Sentence', 'Avg.R_Anticipation']]
    test_Surprise = df.loc[df['Train/Div/Test'].isin(['test']),['UserID', 'Sentence', 'Avg.R_Surprise']]
    test_Anger = df.loc[df['Train/Div/Test'].isin(['test']),['UserID', 'Sentence', 'Avg.R_Anger']]
    test_Fear = df.loc[df['Train/Div/Test'].isin(['test']),['UserID', 'Sentence', 'Avg.R_Fear']]
    test_Disgust = df.loc[df['Train/Div/Test'].isin(['test']),['UserID', 'Sentence', 'Avg.R_Disgust']]
    test_Trust = df.loc[df['Train/Div/Test'].isin(['test']),['UserID', 'Sentence', 'Avg.R_Trust']]
    '''

    
    test_Joy = df.loc[df['Train/Div/Test'].isin(['test']),['UserID', 'Sentence', 'W_Joy']]
    test_Sadness = df.loc[df['Train/Div/Test'].isin(['test']),['UserID', 'Sentence', 'W_Sadness']]
    test_Anticipation = df.loc[df['Train/Div/Test'].isin(['test']),['UserID', 'Sentence', 'W_Anticipation']]
    test_Surprise = df.loc[df['Train/Div/Test'].isin(['test']),['UserID', 'Sentence', 'W_Surprise']]
    test_Anger = df.loc[df['Train/Div/Test'].isin(['test']),['UserID', 'Sentence', 'W_Anger']]
    test_Fear = df.loc[df['Train/Div/Test'].isin(['test']),['UserID', 'Sentence', 'W_Fear']]
    test_Disgust = df.loc[df['Train/Div/Test'].isin(['test']),['UserID', 'Sentence', 'W_Disgust']]
    test_Trust = df.loc[df['Train/Div/Test'].isin(['test']),['UserID', 'Sentence', 'W_Trust']]
    

    train.reset_index(drop=True, inplace=True)
    dev.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    uids_train = [[v] for v in list(train['UserID'])]
    uids_dev = [[v] for v in list(dev['UserID'])]
    uids_test = [[v] for v in list(test['UserID'])]

    
    test_Joy.reset_index(drop=True, inplace=True)
    test_Sadness.reset_index(drop=True, inplace=True)
    test_Anticipation.reset_index(drop=True, inplace=True)
    test_Surprise.reset_index(drop=True, inplace=True)
    test_Anger.reset_index(drop=True, inplace=True)
    test_Fear.reset_index(drop=True, inplace=True)
    test_Disgust.reset_index(drop=True, inplace=True)
    test_Trust.reset_index(drop=True, inplace=True)
    

    y_train = [[v] for v in list(train[emotion1])]
    y_dev = [[v] for v in list(dev[emotion1])]
    y_test = [[v] for v in list(test[emotion1])]

    '''
    y_test_Joy = [[v] for v in list(test_Joy['Avg.R_Joy'])]
    y_test_Sadness = [[v] for v in list(test_Sadness['Avg.R_Sadness'])]
    y_test_Anticipation = [[v] for v in list(test_Anticipation['Avg.R_Anticipation'])]
    y_test_Surprise = [[v] for v in list(test_Surprise['Avg.R_Surprise'])]
    y_test_Anger = [[v] for v in list(test_Anger['Avg.R_Anger'])]
    y_test_Fear = [[v] for v in list(test_Fear['Avg.R_Fear'])]
    y_test_Disgust = [[v] for v in list(test_Disgust['Avg.R_Disgust'])]
    y_test_Trust = [[v] for v in list(test_Trust['Avg.R_Trust'])]
    '''
    
   
    y_test_Joy = [[v] for v in list(test_Joy['W_Joy'])]
    y_test_Sadness = [[v] for v in list(test_Sadness['W_Sadness'])]
    y_test_Anticipation = [[v] for v in list(test_Anticipation['W_Anticipation'])]
    y_test_Surprise = [[v] for v in list(test_Surprise['W_Surprise'])]
    y_test_Anger = [[v] for v in list(test_Anger['W_Anger'])]
    y_test_Fear = [[v] for v in list(test_Fear['W_Fear'])]
    y_test_Disgust = [[v] for v in list(test_Disgust['W_Disgust'])]
    y_test_Trust = [[v] for v in list(test_Trust['W_Trust'])]
    

    pretrained = 'cl-tohoku/bert-base-japanese-whole-word-masking'

    tokenizer = BertJapaneseTokenizer.from_pretrained(pretrained)

    dataset_train = CreateDataset(train['Sentence'], y_train, uids_train, tokenizer, MAX_LEN)
    dataset_dev = CreateDataset(dev['Sentence'], y_dev, uids_dev, tokenizer, MAX_LEN)
    dataset_test = CreateDataset(test['Sentence'], y_test, uids_test, tokenizer, MAX_LEN)

    
    dataset_test_Joy = CreateDataset(test_Joy['Sentence'], y_test_Joy, uids_test, tokenizer, MAX_LEN)
    dataset_test_Sadness = CreateDataset(test_Sadness['Sentence'], y_test_Sadness, uids_test, tokenizer, MAX_LEN)
    dataset_test_Anticipation = CreateDataset(test_Anticipation['Sentence'], y_test_Anticipation, uids_test, tokenizer, MAX_LEN)
    dataset_test_Surprise = CreateDataset(test_Surprise['Sentence'], y_test_Surprise, uids_test, tokenizer, MAX_LEN)
    dataset_test_Anger = CreateDataset(test_Anger['Sentence'], y_test_Anger, uids_test, tokenizer, MAX_LEN)
    dataset_test_Fear = CreateDataset(test_Fear['Sentence'], y_test_Fear, uids_test, tokenizer, MAX_LEN)
    dataset_test_Disgust = CreateDataset(test_Disgust['Sentence'], y_test_Disgust, uids_test, tokenizer, MAX_LEN)
    dataset_test_Trust = CreateDataset(test_Trust['Sentence'], y_test_Trust, uids_test, tokenizer, MAX_LEN)
    

    model = BERTClass(pretrained, DROP_RATE, OUTPUT_SIZE)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    device = 'cuda' if cuda.is_available() else 'cpu'

    mlp_feature = torch.load('test.pt')

    #log = train_model(dataset_train, dataset_dev, BATCH_SIZE, model, criterion, optimizer, NUM_EPOCHS, model_path, mlp_feature, device=device)
    
    # テスト開始
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

    
    dataloader_test_Joy = DataLoader(dataset_test_Joy, batch_size=1, shuffle=False)
    dataloader_test_Sadness = DataLoader(dataset_test_Sadness, batch_size=1, shuffle=False)
    dataloader_test_Anticipation = DataLoader(dataset_test_Anticipation, batch_size=1, shuffle=False)
    dataloader_test_Surprise = DataLoader(dataset_test_Surprise, batch_size=1, shuffle=False)
    dataloader_test_Anger = DataLoader(dataset_test_Anger, batch_size=1, shuffle=False)
    dataloader_test_Fear = DataLoader(dataset_test_Fear, batch_size=1, shuffle=False)
    dataloader_test_Disgust = DataLoader(dataset_test_Disgust, batch_size=1, shuffle=False)
    dataloader_test_Trust = DataLoader(dataset_test_Trust, batch_size=1, shuffle=False)
    
    
    lst1 = []
    lst2 = []
#for model_path, dataloader_test in [(model_path, dataloader_test)]:
#for model_path, dataloader_test in [('models/tohoku-BERT_W_Joy.pth', dataloader_test_Joy), ('models/tohoku-BERT_W_Sadness.pth', dataloader_test_Sadness), ('models/tohoku-BERT_W_Anticipation.pth', dataloader_test_Anticipation), ('models/tohoku-BERT_W_Surprise.pth', dataloader_test_Surprise), ('models/tohoku-BERT_W_Anger.pth', dataloader_test_Anger), ('models/tohoku-BERT_W_Fear.pth', dataloader_test_Fear), ('models/tohoku-BERT_W_Disgust.pth', dataloader_test_Disgust), ('models/tohoku-BERT_W_Trust.pth', dataloader_test_Trust)]:
#for model_path, dataloader_test in [('models/tohoku-BERT_R_Joy.pth', dataloader_test_Joy), ('models/tohoku-BERT_R_Sadness.pth', dataloader_test_Sadness), ('models/tohoku-BERT_R_Anticipation.pth', dataloader_test_Anticipation), ('models/tohoku-BERT_R_Surprise.pth', dataloader_test_Surprise), ('models/tohoku-BERT_R_Anger.pth', dataloader_test_Anger), ('models/tohoku-BERT_R_Fear.pth', dataloader_test_Fear), ('models/tohoku-BERT_R_Disgust.pth', dataloader_test_Disgust), ('models/tohoku-BERT_R_Trust.pth', dataloader_test_Trust)]:
#for model_path, dataloader_test in [('models/tohoku-Concat_W_Joy.pth', dataloader_test_Joy), ('models/tohoku-Concat_W_Sadness.pth', dataloader_test_Sadness), ('models/tohoku-Concat_W_Anticipation.pth', dataloader_test_Anticipation), ('models/tohoku-Concat_W_Surprise.pth', dataloader_test_Surprise), ('models/tohoku-Concat_W_Anger.pth', dataloader_test_Anger), ('models/tohoku-Concat_W_Fear.pth', dataloader_test_Fear), ('models/tohoku-Concat_W_Disgust.pth', dataloader_test_Disgust), ('models/tohoku-Concat_W_Trust.pth', dataloader_test_Trust)]:
    for model_path, dataloader_test in [('models/tohoku-Product_W_Joy.pth', dataloader_test_Joy), ('models/tohoku-Product_W_Sadness.pth', dataloader_test_Sadness), ('models/tohoku-Product_W_Anticipation.pth', dataloader_test_Anticipation), ('models/tohoku-Product_W_Surprise.pth', dataloader_test_Surprise), ('models/tohoku-Product_W_Anger.pth', dataloader_test_Anger), ('models/tohoku-Product_W_Fear.pth', dataloader_test_Fear), ('models/tohoku-Product_W_Disgust.pth', dataloader_test_Disgust), ('models/tohoku-Product_W_Trust.pth', dataloader_test_Trust)]:
        model.load_state_dict(torch.load(early_stopping.path))
        model.to(device)
        model.eval()

        y_pred = []
        y_true = []
        for data in dataloader_test:
            uids = data['uids'].to(device)
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            labels = torch.flatten(data['labels'].to(device))

            outputs = model(ids, mask, uids, mlp_feature, device, mode=MODE)
        
            y_pred.append(torch.argmax(outputs, dim=-1).cpu().numpy())
            y_true.append(labels.cpu().numpy())

        print(model_path)
        print('ACC:', accuracy_score(y_true, y_pred))
        print('MAE:', mean_absolute_error(y_true, y_pred))
        print('QWK:', cohen_kappa_score(y_true, y_pred, weights='quadratic'))

        lst1 += y_pred
        lst2 += y_true

    print('Overall')
    print('ACC:', format(accuracy_score(lst2, lst1), '.3f'))
    print('MAE:', format(mean_absolute_error(lst2, lst1), '.3f'))
    print('QWK:', format(cohen_kappa_score(lst2, lst1, weights='quadratic'), '.3f'))
    

if __name__ == "__main__":
    main()


