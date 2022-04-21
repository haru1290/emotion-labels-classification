import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.optim as optim
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel
from transformers.tokenization_bert_japanese import BertJapaneseTokenizer


TEXT_COLUMN = 'Sentence'
LABEL_COLUMN = 'W_Joy'

BERT_MODEL = 'cl-tohoku/bert-base-japanese-whole-word-masking'

N_EPOCHS = 3
MAX_TOKEN_LEN = 128
OUTPUT_SIZE = 4
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
        text = data_row[TEXT_COLUMN]
        labels = data_row[LABEL_COLUMN]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return dict(
            text=text,
            input_ids=encoding['input_ids'].flatten(),
            attention_mask=encoding['attention_mask'].flatten(),
            labels=torch.tensor(labels)
        )


class CreateDataModule(pl.LightningDataModule):
    def __init__(self, train_df, valid_df, test_df, batch_size=BATCH_SIZE, max_token_len=MAX_TOKEN_LEN, pretrained_model=BERT_MODEL):
        super().__init__()
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

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=os.cpu_count())

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=os.cpu_count())


class EmotionClassifier(pl.LightningModule):
    def __init__(self, n_classes: int, n_epochs=None, drop_rate=None, pretrained_model=BERT_MODEL):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model, return_dict=True)
        self.drop = nn.Dropout(drop_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.n_epochs = n_epochs
        self.criterion = nn.CrossEntropyLoss()

        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.encoder.layer[-1].parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        preds = self.classifier(self.drop(output.pooler_output))
        loss = 0
        if labels is not None:
            loss = self.criterion(preds, labels)

        return loss, preds

    def training_step(self, batch, batch_idx):
        loss, preds = self.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )

        return {
            'loss': loss,
            'batch_preds': preds,
            'batch_labels': batch['labels']
        }

    def validation_step(self, batch, batch_idx):
        loss, preds = self.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )

        return {
            'loss': loss,
            'batch_preds': preds,
            'batch_labels': batch['labels']
        }

    def test_step(self, batch, batch_idx):
        loss, preds = self.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )

        return {
            'loss': loss,
            'batch_preds': preds,
            'batch_labels': batch['labels']
        }

    def validation_epoch_end(self, outputs, mode='val'):
        # Loss Calculation
        epoch_preds = torch.cat([x['batch_preds'] for x in outputs])
        epoch_labels = torch.cat([x['batch_labels'] for x in outputs])
        epoch_loss = self.criterion(epoch_preds, epoch_labels)
        self.log(f"{mode}_loss", epoch_loss, logger=True)

        # Accuracy Calculation
        num_correct = (epoch_preds.argmax(dim=1) == epoch_labels).sum().item()
        epoch_accuracy = num_correct / len(epoch_labels)
        self.log(f"{mode}_accuracy", epoch_accuracy, logger=True)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, 'test')

    def configure_optimizers(self):
        optimizer = optim.Adam([
            {'params': self.bert.encoder.layer[-1].parameters(), 'lr': LEARNING_RATE},
            {'params': self.classifier.parameters(), 'lr': LEARNING_RATE}
        ])

        return [optimizer]

    
def main():
    df = pd.read_csv("./data/pn-long.csv", header=0)
    
    train_df = df[df['Train/Div/Test'] == 'train'].reset_index(drop=True)
    valid_df = df[df['Train/Div/Test'] == 'dev'].reset_index(drop=True)
    test_df = df[df['Train/Div/Test'] == 'test'].reset_index(drop=True)

    data_module = CreateDataModule(train_df, valid_df, test_df)
    data_module.setup()
    
    model = EmotionClassifier(n_classes=OUTPUT_SIZE, n_epochs=N_EPOCHS, drop_rate=DROP_RATE)

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.05,
        patience=3,
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints",
        filename="{epoch}",
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    trainer = pl.Trainer(
        max_epochs=N_EPOCHS,
        gpus=1,
        progress_bar_refresh_rate=30,
        callbacks=[checkpoint_callback, early_stop_callback]
    )

    trainer.fit(model, data_module)

    result = trainer.test(ckpt_path=checkpoint_callback.best_model_path)


if __name__ == '__main__':
    main()
