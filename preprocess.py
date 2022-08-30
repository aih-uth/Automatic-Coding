import os, re, json
import pandas as pd
import json, MeCab
from pathlib import Path
from transformers import BertTokenizer, BertModel
import jaconv, unicodedata, regex, argparse
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import mojimoji
import logging
import sys
import pandas as pd
import mojimoji

from lib.util import read_medis, read_manbyo, read_t, mecab_wakati, preprocess


def main():
    # データの読み込み
    medis_df, _ = read_medis()
    manbyo_df = read_manbyo()
    # ベースとなる学習データの作成
    train_df = expand_medis(medis_df, manbyo_df)
    # 形態素解析
    train_df["token"] = wakati(train_df["word"])
    train_df.to_csv("./data/train/train.csv", index=False)
    # データ拡張する場合
    if args.data_aug:
        # ティ辞書の読み込み
        t_df = read_t()
        t_df["token"] = wakati(t_df["term"])
        t_df.to_csv("./data/train/t_df.csv", index=False)


def expand_medis(medis_df, manbyo_df):
    # MEDISを万病辞書で拡張する
    dct = {"word": [], "medis": [], "icd": []}
    manbyo_vocab = {x: 0 for x in manbyo_df["標準病名"]}
    for w, m, i in zip(medis_df["word"], medis_df["medis"], medis_df["icd"]):
        if w in manbyo_vocab:
            tmp_df = manbyo_df[manbyo_df["標準病名"]==w]
            for iw in tmp_df["出現形"]:
                # 出現形と標準病名が異なる場合は追加
                if iw != w:
                    dct["word"].append(iw)
                    dct["medis"].append(m)
                    dct["icd"].append(i)
    ex_manbyo_df = df = pd.DataFrame.from_dict(dct, orient='index').T
    # 縦につなげて学習データとします
    medis_df["source"] = "medis"
    ex_manbyo_df["source"] = "manbyo"
    train_df = pd.concat([medis_df, ex_manbyo_df])
    return train_df


def wakati(words):
    tokens = []
    for w in words:
        sub_words = tokenizer.tokenize(" ".join(mecab_wakati(mecab, preprocess(w))))
        tokens.append(" ".join(sub_words))
    return tokens


if __name__ == '__main__':
    # 引数
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_path', type=str, default='../BERT/UTH_BERT_BASE_512_MC_BPE_WWM_V25000_352K')
    parser.add_argument('--manbyo_path', type=str, default="./data/manbyo/MANBYO_201907_Dic-utf8.dic")
    parser.add_argument('--neologd_path', type=str, default="/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")
    # 与えなければTrue
    parser.add_argument('--data_aug', action='store_false')

    args = parser.parse_args()

    mecab = MeCab.Tagger("-d {0} -u {1}".format(args.neologd_path, args.manbyo_path))
    tokenizer = BertTokenizer(Path(args.bert_path) / "vocab.txt", do_lower_case=False, do_basic_tokenize=False)

    main()
        