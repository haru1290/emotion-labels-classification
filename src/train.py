import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments, BertForSequenceClassification, BertTokenizer
from sklearn.metrics import accuracy_score, mean_absolute_error, cohen_kappa_score
from argparse import ArgumentParser

'''from make_dataset import CreateDataModule
from models import BertClassifier'''


PATIENCE = 3
BEST_MODEL_PATH = './models/best_model.pth'

BERT_MODEL = 'cl-tohoku/bert-base-japanese-whole-word-masking'

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
        data_row = self.data[index]
        text = data_row['Sentence']
        label = data_row['Writer_Joy']

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            pad_to_max_length=True,
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

    print(labels.shape, preds.shape)

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
    tokenizer = BertTokenizer.from_pretrained(pretrained)

    train_dataset = CreateDataset(train, tokenizer, MAX_TOKEN_LEN)
    eval_dataset = CreateDataset(eval, tokenizer, MAX_TOKEN_LEN)
    test_dataset = CreateDataset(test, tokenizer, MAX_TOKEN_LEN)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        dataloader_drop_last=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()


    '''criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam([
        {'params': model.module.bert.parameters(), 'lr': LEARNING_RATE},
        {'params': model.module.classifier.parameters(), 'lr': LEARNING_RATE}
    ])
    
    earlystopping = EarlyStopping(
        patience=PATIENCE,
        verbose=True
    )

    train_step(model, data_module.train_dataloader(), data_module.valid_dataloader(), criterion, optimizer, earlystopping, device, n_epochs=N_EPOCHS)

    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    test_scores = calculate_loss_and_scores(model, data_module.test_dataloader(), criterion, device)
    print(f"[Test] ACC: {test_scores['accuracy_score']:.3f}, MAE: {test_scores['mean_absolute_error']:.3f}, QWK: {test_scores['cohen_kappa_score']:.3f}")
'''

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--train')
    parser.add_argument('--valid')
    parser.add_argument('--test')

    args = parser.parse_args()

    main(args)
