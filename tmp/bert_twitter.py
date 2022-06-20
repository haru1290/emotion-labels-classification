import os, sys
import itertools
import numpy as np
import pandas as pd
import transformers
import tensorflow as tf
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertForPreTraining, BertTokenizer
from torch import optim
from torch import cuda
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import cohen_kappa_score

from earlystopping import EarlyStopping

sys.path.append("./hottoSNS-bert/src/")
import tokenization
from preprocess import normalizer

MODE = None

MAX_LEN = 128
DROP_RATE = 0.1
HIDDEN_SIZE = 768
OUTPUT_SIZE = 4
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 2e-5
PATIENCE = 3


class CreateDataset(Dataset):
    def __init__(self, uids, X, y, tokenizer, max_len):
        self.uids = uids
        self.X = X
        self.y = y
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        uids = self.uids[index]
        text = self.X[index]
        inputs = self.tokenizer.tokenize(text)
        ids = self.tokenizer.convert_tokens_to_ids(["[CLS]"]+inputs+["[SEP]"])
        mask = [1] * len(ids)
        while self.max_len > len(ids):
            ids.append(0)
            mask.append(0)
        if self.max_len < len(ids):
            ids = ids[:self.max_len]
            mask = mask[:self.max_len]

        return {
            'text': text,
            'uids': torch.LongTensor(uids),
            'ids': torch.LongTensor(ids),
            'mask': torch.LongTensor(mask),
            'labels': torch.LongTensor(self.y[index])
        }


class BERTClass(torch.nn.Module):
    def __init__(self, bert_model_file, drop_rate, hidden_size, output_size, config_file):
        super().__init__()
        self.bert = BertForPreTraining.from_pretrained(bert_model_file, from_tf=True, config=config_file, output_hidden_states=True)
        self.drop = torch.nn.Dropout(drop_rate)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, ids, mask, uids, mlp_feature, mode=None):
        output = self.bert(ids, attention_mask=mask)[2][-1][:,0]

        '''lst = torch.empty(len(output), 768).to('cuda:0')
        if mode == 'product':
            for i in range(len(output)):
                lst[i] = output[i] * mlp_feature[uids[i][0].item()-1]
            output = lst
        elif mode == 'concat':
            for i in range(len(output)):
                lst[i] = torch.cat((output[i], mlp_feature[uids[i][0].item()-1]), 0)
            output = lst'''

        output = self.fc(self.drop(output))

        return output


def train_model(dataset_train, dataset_valid, batch_size, model, criterion, optimizer, earlystopping, num_epochs, mlp_feature, model_path, device=None):
    model.to(device)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        # 学習
        train_loss = 0
        train_acc = 0
        train_total = 0
        model.train()

        for data in dataloader_train:
            uids = data['uids'].to(device)
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            labels = torch.flatten(data['labels'].to(device))

            optimizer.zero_grad()

            output = model(ids, mask, uids, mlp_feature, mode=MODE)

            loss = criterion(output, labels)
            
            pred = torch.argmax(output, dim=-1).cpu().numpy()
            labels = labels.cpu().numpy()

            train_loss += loss.item()
            train_acc += (pred == labels).sum().item()
            train_total += len(labels)

            loss.backward()
            optimizer.step()
        
        train_loss = train_loss / train_total
        train_acc = train_acc / train_total
        
        # 推論
        valid_loss = 0
        valid_acc = 0
        valid_total = 0
        model.eval()

        for data in dataloader_valid:
            uids = data['uids'].to(device)
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            labels = torch.flatten(data['labels'].to(device))

            with torch.no_grad():
                output = model(ids, mask, uids, mlp_feature, mode=MODE)
            
            loss = criterion(output, labels)
            
            pred = torch.argmax(output, dim=-1).cpu().numpy()
            labels = labels.cpu().numpy()
            
            valid_loss += loss.item()
            valid_acc += (pred == labels).sum().item()
            valid_total += len(labels)
        
        valid_loss = valid_loss / valid_total
        valid_acc = valid_acc / valid_total

        earlystopping(valid_loss, model)
        if earlystopping.early_stop:
            print("Early stopping")
            break

    
def main():
    df = pd.read_csv('../data/tsv.csv', header=0)

    emotion1 = 'W_Joy'
    #model_path = args[2]
    
    train = df.loc[df['Train/Div/Test'].isin(['train']),['UserID', 'Sentence', emotion1]]
    valid = df.loc[df['Train/Div/Test'].isin(['dev']),['UserID', 'Sentence', emotion1]]
    test = df.loc[df['Train/Div/Test'].isin(['test']),['UserID', 'Sentence', emotion1]]

    
    '''train_Joy = df.loc[df['Train/Div/Test'].isin(['train']),['UserID', 'Sentence', 'Avg.R_Joy']]
    train_Sadness = df.loc[df['Train/Div/Test'].isin(['train']),['UserID', 'Sentence', 'Avg.R_Sadness']]
    train_Anticipation = df.loc[df['Train/Div/Test'].isin(['train']),['UserID', 'Sentence', 'Avg.R_Anticipation']]
    train_Surprise = df.loc[df['Train/Div/Test'].isin(['train']),['UserID', 'Sentence', 'Avg.R_Surprise']]
    train_Anger = df.loc[df['Train/Div/Test'].isin(['train']),['UserID', 'Sentence', 'Avg.R_Anger']]
    train_Fear = df.loc[df['Train/Div/Test'].isin(['train']),['UserID', 'Sentence', 'Avg.R_Fear']]
    train_Disgust = df.loc[df['Train/Div/Test'].isin(['train']),['UserID', 'Sentence', 'Avg.R_Disgust']]
    train_Trust = df.loc[df['Train/Div/Test'].isin(['train']),['UserID', 'Sentence', 'Avg.R_Trust']]'''
    
    train.reset_index(drop=True, inplace=True)
    valid.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    
    '''train_Joy.reset_index(drop=True, inplace=True)
    train_Sadness.reset_index(drop=True, inplace=True)
    train_Anticipation.reset_index(drop=True, inplace=True)
    train_Surprise.reset_index(drop=True, inplace=True)
    train_Anger.reset_index(drop=True, inplace=True)
    train_Fear.reset_index(drop=True, inplace=True)
    train_Disgust.reset_index(drop=True, inplace=True)
    train_Trust.reset_index(drop=True, inplace=True)'''

    uids_train = [[v] for v in list(train['UserID'])]
    uids_valid = [[v] for v in list(valid['UserID'])]
    uids_test = [[v] for v in list(test['UserID'])]

    '''uids_train_Joy = [[v] for v in list(train_Joy['UserID'])]
    uids_train_Sadness = [[v] for v in list(train_Sadness['UserID'])]
    uids_train_Anticipation = [[v] for v in list(train_Anticipation['UserID'])]
    uids_train_Surprise = [[v] for v in list(train_Surprise['UserID'])]
    uids_train_Anger = [[v] for v in list(train_Anger['UserID'])]
    uids_train_Fear = [[v] for v in list(train_Fear['UserID'])]
    uids_train_Disgust = [[v] for v in list(train_Disgust['UserID'])]
    uids_train_Trust = [[v] for v in list(train_Trust['UserID'])]'''
    
    y_train = [[v] for v in list(train[emotion1])]
    y_valid = [[v] for v in list(valid[emotion1])]
    y_test = [[v] for v in list(test[emotion1])]

    '''y_train_Joy = [[v] for v in list(train_Joy['Avg.R_Joy'])]
    y_train_Sadness = [[v] for v in list(train_Sadness['Avg.R_Sadness'])]
    y_train_Anticipation = [[v] for v in list(train_Anticipation['Avg.R_Anticipation'])]
    y_train_Surprise = [[v] for v in list(train_Surprise['Avg.R_Surprise'])]
    y_train_Anger = [[v] for v in list(train_Anger['Avg.R_Anger'])]
    y_train_Fear = [[v] for v in list(train_Fear['Avg.R_Fear'])]
    y_train_Disgust = [[v] for v in list(train_Disgust['Avg.R_Disgust'])]
    y_train_Trust = [[v] for v in list(train_Trust['Avg.R_Trust'])]'''
    
    bert_model_dir = "./hottoSNS-bert/trained_model/masked_lm_only_L-12_H-768_A-12/"
    config_file = os.path.join(bert_model_dir, "bert_config.json")
    vocab_file = os.path.join(bert_model_dir, "tokenizer_spm_32K.vocab.to.bert")
    sp_model_file = os.path.join(bert_model_dir, "tokenizer_spm_32K.model")
    bert_model_file = os.path.join(bert_model_dir, "model.ckpt-1000000.index")
    
    tokenizer = tokenization.JapaneseTweetTokenizer(
        vocab_file = vocab_file,
        model_file = sp_model_file,
        normalizer = normalizer.twitter_normalizer_for_bert_encoder,
        do_lower_case = False
    )

    dataset_train = CreateDataset(uids_train, train['Sentence'], y_train, tokenizer, MAX_LEN)
    dataset_valid = CreateDataset(uids_valid, valid['Sentence'], y_valid, tokenizer, MAX_LEN)
    dataset_test = CreateDataset(uids_test, test['Sentence'], y_test, tokenizer, MAX_LEN)
    
    '''dataset_train_Joy = CreateDataset(uids_train_Joy, train_Joy['Sentence'], y_train_Joy, tokenizer, MAX_LEN)
    dataset_train_Sadness = CreateDataset(uids_train_Sadness, train_Sadness['Sentence'], y_train_Sadness, tokenizer, MAX_LEN)
    dataset_train_Anticipation = CreateDataset(uids_train_Anticipation, train_Anticipation['Sentence'], y_train_Anticipation, tokenizer, MAX_LEN)
    dataset_train_Surprise = CreateDataset(uids_train_Surprise, train_Surprise['Sentence'], y_train_Surprise, tokenizer, MAX_LEN)
    dataset_train_Anger = CreateDataset(uids_train_Anger, train_Anger['Sentence'], y_train_Anger, tokenizer, MAX_LEN)
    dataset_train_Fear = CreateDataset(uids_train_Fear, train_Fear['Sentence'], y_train_Fear, tokenizer, MAX_LEN)
    dataset_train_Disgust = CreateDataset(uids_train_Disgust, train_Disgust['Sentence'], y_train_Disgust, tokenizer, MAX_LEN)
    dataset_train_Trust = CreateDataset(uids_train_Trust, train_Trust['Sentence'], y_train_Trust, tokenizer, MAX_LEN)'''

    model = BERTClass(bert_model_file, DROP_RATE, HIDDEN_SIZE, OUTPUT_SIZE, config_file)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    device = 'cuda' if cuda.is_available() else 'cpu'

    mlp_feature = torch.load('test.pt')

    earlystopping = EarlyStopping(
        patience=PATIENCE,
        verbose=True
    )

    # train_model(dataset_train, dataset_valid, BATCH_SIZE, model, criterion, optimizer, earlystopping, NUM_EPOCHS, mlp_feature, model_path, device=device)
    
    # テスト
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)
    
    '''dataloader_train_Joy = DataLoader(dataset_train_Joy, batch_size=1, shuffle=False)
    dataloader_train_Sadness = DataLoader(dataset_train_Sadness, batch_size=1, shuffle=False)
    dataloader_train_Anticipation = DataLoader(dataset_train_Anticipation, batch_size=1, shuffle=False)
    dataloader_train_Surprise = DataLoader(dataset_train_Surprise, batch_size=1, shuffle=False)
    dataloader_train_Anger = DataLoader(dataset_train_Anger, batch_size=1, shuffle=False)
    dataloader_train_Fear = DataLoader(dataset_train_Fear, batch_size=1, shuffle=False)
    dataloader_train_Disgust = DataLoader(dataset_train_Disgust, batch_size=1, shuffle=False)
    dataloader_train_Trust = DataLoader(dataset_train_Trust, batch_size=1, shuffle=False)'''

    lst1 = []
    lst2 = []
    lst3 = []
    #for model_path, dataloader in [('models/tohoku-BERT_R_Joy.pth', dataloader_test_Joy), ('models/tohoku-BERT_R_Sadness.pth', dataloader_test_Sadness), ('models/tohoku-BERT_R_Anticipation.pth', dataloader_test_Anticipation), ('models/tohoku-BERT_R_Surprise.pth', dataloader_test_Surprise), ('models/tohoku-BERT_R_Anger.pth', dataloader_test_Anger), ('models/tohoku-BERT_R_Fear.pth', dataloader_test_Fear), ('models/tohoku-BERT_R_Disgust.pth', dataloader_test_Disgust), ('models/tohoku-BERT_R_Trust.pth', dataloader_test_Trust)]:
    #for model_path, dataloader in [('models/hottoSNS-BERT_W_Joy.pth', dataloader_train_Joy), ('models/hottoSNS-BERT_W_Sadness.pth', dataloader_train_Sadness), ('models/hottoSNS-BERT_W_Anticipation.pth', dataloader_train_Anticipation), ('models/hottoSNS-BERT_W_Surprise.pth', dataloader_train_Surprise), ('models/hottoSNS-BERT_W_Anger.pth', dataloader_train_Anger), ('models/hottoSNS-BERT_W_Fear.pth', dataloader_train_Fear), ('models/hottoSNS-BERT_W_Disgust.pth', dataloader_train_Disgust), ('models/hottoSNS-BERT_W_Trust.pth', dataloader_train_Trust)]:
    #for model_path, dataloader in [('models/hottoSNS-BERT_R_Joy.pth', dataloader_train_Joy), ('models/hottoSNS-BERT_R_Sadness.pth', dataloader_train_Sadness), ('models/hottoSNS-BERT_R_Anticipation.pth', dataloader_train_Anticipation), ('models/hottoSNS-BERT_R_Surprise.pth', dataloader_train_Surprise), ('models/hottoSNS-BERT_R_Anger.pth', dataloader_train_Anger), ('models/hottoSNS-BERT_R_Fear.pth', dataloader_train_Fear), ('models/hottoSNS-BERT_R_Disgust.pth', dataloader_train_Disgust), ('models/hottoSNS-BERT_R_Trust.pth', dataloader_train_Trust)]:
    #for model_path, dataloader in [('models/Concat_W_Joy.pth', dataloader_test_Joy), ('models/Concat_W_Sadness.pth', dataloader_test_Sadness), ('models/Concat_W_Anticipation.pth', dataloader_test_Anticipation), ('models/Concat_W_Surprise.pth', dataloader_test_Surprise), ('models/Concat_W_Anger.pth', dataloader_test_Anger), ('models/Concat_W_Fear.pth', dataloader_test_Fear), ('models/Concat_W_Disgust.pth', dataloader_test_Disgust), ('models/Concat_W_Trust.pth', dataloader_test_Trust)]:
    #for model_path, dataloader in [('models/Product_W_Joy.pth', dataloader_test_Joy), ('models/Product_W_Sadness.pth', dataloader_test_Sadness), ('models/Product_W_Anticipation.pth', dataloader_test_Anticipation), ('models/Product_W_Surprise.pth', dataloader_test_Surprise), ('models/Product_W_Anger.pth', dataloader_test_Anger), ('models/Product_W_Fear.pth', dataloader_test_Fear), ('models/Product_W_Disgust.pth', dataloader_test_Disgust), ('models/Product_W_Trust.pth', dataloader_test_Trust)]:
    for model_path, dataloader in [(model_path, dataloader_test)]:
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()
        
        y_pred = []
        y_true = []
        y_text = []
        for data in dataloader:
            text = data['text']
            uids = data['uids'].to(device)
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            labels = torch.flatten(data['labels'].to(device))

            with torch.no_grad():
                output = model(ids, mask, uids, mlp_feature, mode=MODE)
            output = torch.argmax(output, dim=-1).cpu().numpy()
            labels = labels.cpu().numpy()
        
            y_pred.append(output)
            y_true.append(labels)
            y_text.append(text)

            '''sample_text = "車のタイヤがパンクしてた。。いたずらの可能性が高いんだって。。"
            if sample_text in text:
                idx = text.index(sample_text)
                print(model_path, labels[idx], output[idx])
                continue'''

        print(model_path)
        print('ACC:', format(accuracy_score(y_true, y_pred), '.3f'))
        print('MAE:', format(mean_absolute_error(y_true, y_pred), '.3f'))
        print('QWK:', format(cohen_kappa_score(y_true, y_pred, weights='quadratic'), '.3f'))

        
        '''with open('./writer-labels.csv', 'a', newline='\n', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(y_text)

        lst1 += y_pred
        lst2 += y_true
        lst3 += y_text   

    print('Overall')
    print('ACC:', format(accuracy_score(lst2, lst1), '.3f'))
    print('MAE:', format(mean_absolute_error(lst2, lst1), '.3f'))
    print('QWK:', format(cohen_kappa_score(lst2, lst1, weights='quadratic'), '.3f'))'''


if __name__ == "__main__":
    main()

