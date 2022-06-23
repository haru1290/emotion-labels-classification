import numpy as np
import pandas as pd
from argparse import ArgumentParser


def main(args):
    df = pd.read_csv(args.wrime, header=0, sep='\t')

    df['Writer_Sentiment'] = df['Writer_Sentiment'] + 2
    df['Reader1_Sentiment'] = df['Reader1_Sentiment'] + 2
    df['Reader2_Sentiment'] = df['Reader2_Sentiment'] + 2
    df['Reader3_Sentiment'] = df['Reader3_Sentiment'] + 2
    df['Avg. Readers_Sentiment'] = df['Avg. Readers_Sentiment'] + 2
    
    train = df[df['Train/Dev/Test'] == 'train']
    valid = df[df['Train/Dev/Test'] == 'dev']
    test = df[df['Train/Dev/Test'] == 'test']

    train.reset_index(drop=True, inplace=True)
    valid.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    train.to_csv(args.train, index=False, sep='\t', encoding='utf_8_sig')
    valid.to_csv(args.valid, index=False, sep='\t', encoding='utf_8_sig')
    test.to_csv(args.test, index=False, sep='\t', encoding='utf_8_sig')


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--wrime', default='./data/wrime-ver2.tsv')
    parser.add_argument('--train', default='./data/train.tsv')
    parser.add_argument('--valid', default='./data/valid.tsv')
    parser.add_argument('--test', default='./data/test.tsv')

    args = parser.parse_args()

    main(args)