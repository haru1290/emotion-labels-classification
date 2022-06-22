import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, BertForSequenceClassification, BertJapaneseTokenizer
from sklearn.metrics import accuracy_score, mean_absolute_error, cohen_kappa_score
from argparse import ArgumentParser


BEST_MODEL_PATH = './models/best_model.pth'

N_EPOCHS = 100
N_CLASSES = 4
MAX_TOKEN_LEN = 128
BATCH_SIZE = 32
DROP_RATE = 0.1
LEARNING_RATE = 2e-5


class CreateDataset(Dataset):
    def __init__(self, data, tokenizer, max_token_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_row = self.data.iloc[index]
        text = data_row['Sentence']
        labels = data_row['Writer_Joy']

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels)
        }
    

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    return {
        'accuracy': accuracy_score(preds, labels),
        'mean_absolute_error': mean_absolute_error(preds, labels),
        'cohen_kappa_score': cohen_kappa_score(preds, labels, weights='quadratic')
    }


def main(args):
    train = pd.read_csv('data/train.tsv', header=0, sep='\t')
    eval = pd.read_csv('data/valid.tsv', header=0, sep='\t')
    test = pd.read_csv('data/test.tsv', header=0, sep='\t')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    pretrained = 'cl-tohoku/bert-base-japanese-whole-word-masking'
    model = BertForSequenceClassification.from_pretrained(pretrained, num_labels=4)
    model = model.to(device)
    tokenizer = BertJapaneseTokenizer.from_pretrained(pretrained)

    train_dataset = CreateDataset(train, tokenizer, 512)
    eval_dataset = CreateDataset(eval, tokenizer, 512)
    test_dataset = CreateDataset(test, tokenizer, 512)

    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        num_train_epochs=100,
        logging_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='cohen_kappa_score'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()
    trainer.evaluate(eval_dataset=test_dataset)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--train')
    parser.add_argument('--valid')
    parser.add_argument('--test')

    args = parser.parse_args()

    main(args)
