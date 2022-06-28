import numpy as np
import pandas as pd
import MeCab
from argparse import ArgumentParser
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, cohen_kappa_score
from tqdm import tqdm

import torch.nn as nn


class LogRegClass:
    def __init__(self):
        self.model = None

    def fit(self, X_train, y_train, C):
        self.model = LogisticRegression(C=C, random_state=0)
        self.model.fit(X_train, y_train)

        return self.model

    def predict(self, X):
        pred = self.model.predict(X)

        return pred


def tokenizer(text):
    tagger = MeCab.Tagger("-Owakati")
    words = tagger.parse(text).split()
    
    return ' '.join(words)


def calculate_score(preds, labels):
    accuracy = accuracy_score(preds, labels)
    mae = mean_absolute_error(preds, labels)
    qwk = cohen_kappa_score(preds, labels, weights='quadratic')

    return {
        'accuracy': round(accuracy, 3),
        'mae': round(mae, 3),
        'qwk': round(qwk, 3)
    }


def main(args):
    train = pd.read_csv(args.train, header=0, sep='\t')
    valid = pd.read_csv(args.valid, header=0, sep='\t')
    test = pd.read_csv(args.test, header=0, sep='\t')

    train_valid = pd.concat([train, valid], axis=0)
    train_valid.reset_index(drop=True, inplace=True)

    vectorizer = CountVectorizer()
    X_train_valid = vectorizer.fit_transform(train_valid['Sentence'])
    X_test = vectorizer.transform(test['Sentence'])

    X_train_valid = pd.DataFrame(X_train_valid.toarray(), columns=vectorizer.get_feature_names())
    X_test = pd.DataFrame(X_test.toarray(), columns=vectorizer.get_feature_names())

    X_train = X_train_valid[:len(train)]
    X_valid = X_train_valid[len(train):]

    print(X_train.shape)
    print(X_valid.shape)
    print(X_test.shape)

    results = []
    for C in tqdm([0.01, 0.1, 1, 10, 100]):
        model = LogRegClass()
        model.fit(X_train, train['Writer_Sentiment'], C)
        valid_preds = model.predict(X_valid)
        test_preds = model.predict(X_test)

        valid_score = calculate_score(valid_preds, valid['Writer_Sentiment'])
        test_score = calculate_score(test_preds, test['Writer_Sentiment'])

        results.append(f'C:{C}, v_acc:{valid_score["accuracy"]}, t_acc:{test_score["accuracy"]}, v_mae:{valid_score["mae"]}, t_mae:{test_score["mae"]}, v_qwk:{valid_score["qwk"]}, t_qwk:{test_score["qwk"]}')
    
    for result in results:
        print(result)


if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument('--train', default='./data/train.tsv')
    parser.add_argument('--valid', default='./data/valid.tsv')
    parser.add_argument('--test', default='./data/test.tsv')

    args = parser.parse_args()
    main(args)
