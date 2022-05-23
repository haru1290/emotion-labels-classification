import numpy as np
import pandas as pd
from argparse import ArgumentParser


def main(args):
    df = pd.read_csv(args.wrime, header=0, sep='\t')
    
    train = df[df['Train/Dev/Test'] == 'train']
    valid = df[df['Train/Dev/Test'] == 'dev']
    test = df[df['Train/Dev/Test'] == 'test']

    train.reset_index(drop=True, inplace=True)
    valid.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    train.to_csv('./data/train.tsv', index=False, sep='\t', encoding='utf_8_sig')
    valid.to_csv('./data/valid.tsv', index=False, sep='\t', encoding='utf_8_sig')
    test.to_csv('./data/test.tsv', index=False, sep='\t', encoding='utf_8_sig')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--wrime', default='./data/wrime-ver1.tsv')
    args = parser.parse_args()

    main(args)