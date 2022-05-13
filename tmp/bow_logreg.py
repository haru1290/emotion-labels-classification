import numpy as np
import pandas as pd
import MeCab
from argparse import ArgumentParser
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, cohen_kappa_score
from tqdm import tqdm


RANDOM_STATE = 34


class LogRegClass:
    def __init__(self):
        self.model = None

    def fit(self, X_train, y_train, C):
        self.model = LogisticRegression(C=C, random_state=RANDOM_STATE)
        self.model.fit(X_train, y_train)

        return model

    def predict(self, X):
        pred = self.model.predict(x)

        return predict


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
    df = pd.read_csv(args.corpus, header=0)
    df['Sentence'] = df['Sentence'].map(lambda x: tokenizer(x))

    train = df[df['Train/Div/Test'] == 'train'].reset_index(drop=True)
    valid = df[df['Train/Div/Test'] == 'dev'].reset_index(drop=True)
    test = df[df['Train/Div/Test'] == 'test'].reset_index(drop=True)

    train_valid = pd.concat([train, valid], axis=0)
    train_valid.reset_index(drop=True, inplace=True)

    vectorizer = CountVectorizer()
    X_train_valid = vectorizer.fit_transform(train_valid['Sentence'])
    X_test = vectorizer.transform(test['Sentence'])

    X_train_valid = pd.DataFrame(X_train_valid.toarray(), columns=vectorizer.get_feature_names())
    X_test = pd.DataFrame(X_test.toarray(), columns=vectorizer.get_feature_names())

    X_train = X_train_valid[:len(train)]
    X_valid = X_train_valid[len(train):]

    results = []
    for C in tqdm([0.01, 0.1, 1, 10, 100]):
        lg = train_model(X_train, args.y_train, C)
        valid_preds = lg.predict(X_valid)
        test_preds = lg.predict(X_test)

        valid_score = calculate_score(valid_preds, valid[args.y_valid])
        test_score = calculate_score(test_preds, test[args.y_test])

        results.append(f'C:{C}, v_acc:{valid_score["accuracy"]}, t_acc:{test_score["accuracy"]}, v_mae:{valid_score["mae"]}, t_mae:{test_score["mae"]}, v_qwk:{valid_score["qwk"]}, t_qwk:{test_score["qwk"]}')
    
    for result in results:
        print(result)


if __name__ == "__main__":
    argparser = ArgumentParser()

    argparser.add_argument('--corpus', default='./data/pn-short.csv')
    
    argparser.add_argument('--y_train', default='W_PN')
    argparser.add_argument('--y_valid', default='W_PN')
    argparser.add_argument('--y_test', default='W_PN')

    args = argparser.parse_args()
    main(args)
