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
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
import logging
import sys
import pandas as pd
import collections
import mojimoji


from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class ExperimentDataset():
    def __init__(self, tokens, labels, tokenizer, medis2idx):
        self.tokens = tokens
        self.labels = labels
        self.tokenizer = tokenizer
        self.medis2idx = medis2idx
        
        
    def __getitem__(self, item):
        batch_text = torch.tensor(self.tokenizer.convert_tokens_to_ids(["[CLS]"] + self.tokens[item].split(" ") + ["[SEP]"]))
        batch_label = torch.tensor(self.medis2idx[self.labels[item]])
        return batch_text, batch_label
    
    
    def __len__(self):
        return len(self.tokens)
    
    
def my_collate_fn(batch):
    # https://note.nkmk.me/python-zip-usage-for/
    # https://teratail.com/questions/182610
    tokens, labels = list(zip(*batch))
    return tokens, labels


def preprocess(text, nfkc=False, h2z=True):
    # Normalization Form Compatibility Composition (NFKC)
    if nfkc:
        text = unicodedata.normalize("NFKC", text)
    # full-width characterization
    text = regex.sub(r'(\d)([,])(\d+)', r'\1\3', text)
    text = text.replace(",", "、")
    text = text.replace("，", "、")
    if h2z:
        text = (jaconv.h2z(text, kana=True, digit=True, ascii=True))
    # remove full-width space
    text = text.replace("\u3000", "")
    return text


# BERTの埋め込み表現を使った類似度測定
def mecab_wakati(mecab, sentence):
    wakatis = []
    for ws in mecab.parse(sentence).split('\n'):
        if ws == "EOS":
            break
        else:
            wakatis.append(ws.split("\t")[0])
    return wakatis


def word_to_idx(df, bert_tokenizer):
    vecs = []
    for w, l in zip(df["token"], df["medis"]):
        vecs.append(bert_tokenizer.convert_tokens_to_ids(["[CLS]"] + w.split(" ")[:100] + ["[SEP]"]))
    return vecs


def read_medis(medis_path="./data/medis/nmain504.csv", index_path="./data/medis/index504.txt"):
    # MEDISの辞書
    ma_df = pd.read_csv(medis_path)
    # MEDISのINDEX
    tmp1, tmp2 = [], []
    for line in open(index_path, encoding="cp932"):
        line = line.replace("\"", "").rstrip().split(",")
        tmp1.append(mojimoji.han_to_zen(line[0]))
        tmp2.append(line[1])

    index2word = {}
    for line in open(index_path, encoding="cp932"):
        line = line.replace("\"", "").rstrip().split(",")
        word = mojimoji.han_to_zen(line[0])
        medis = line[1]
        if medis in index2word:
            index2word[medis].append(word)
        else:
            index2word[medis] = [word]

    # INDEXのデータを追加
    tmp1, tmp2, tmp3 = [], [], []
    for w, m, i in zip(ma_df["病名表記"], ma_df["病名交換用コード"], ma_df["ICD10-2013"]):
        tmp1.append(w)
        tmp2.append(m)
        tmp3.append(i)
        if m in index2word:
            for iw in index2word[m]:
                tmp1.append(iw)
                tmp2.append(m)
                tmp3.append(i)

    df = pd.DataFrame({"word": tmp1, "medis": tmp2, "icd": tmp3})
    # 重複を削除
    df = df[~df.duplicated()]

    return df, ma_df


def read_t(t_path="./data/t/tdic_tenkai202111_UTF8.csv"):
    t_df = pd.read_csv(t_path, header=None, names=['num', 'index', 'code', 'term', 'yomi'])
    # 病名のみを取得
    t_df["digit"] = [str(x)[0] for x in t_df["num"]]
    t_df = t_df[t_df["digit"]=="1"].drop("digit", axis=1)
    return t_df


def read_manbyo(manbyo_path="./data/manbyo/MANBYO_20210602.csv"):
    manbyo_df = pd.read_csv(manbyo_path)
    manbyo_df = manbyo_df.query('信頼度レベル in ["S", "A", "B"]')
    manbyo_df = manbyo_df.query('標準病名 != "-1"')
    return manbyo_df


def data_augmentation(X_train, t_df):
    # データ拡張をする場合の処理: ティ辞書から同義語を得る
    t_dct_vocab = {w: 0 for w in t_df["term"]}
    # dct = {"word": [], "medis": [], "icd": [], "source": [], "token": []}
    dct = {"word": [], "medis": [], "source": [], "token": []}
    # for word, medis, icd in zip(X_train["word"], X_train["medis"], X_train["icd"]):
    for word, medis in zip(X_train["word"], X_train["medis"]):
        if word in t_dct_vocab:
            # ティ辞書を検索
            tmp_df = t_df.query("@word in term")
            if 11 in {n: 0 for n in tmp_df["code"].to_list()}:
                # ティ辞書内で同義語として登録されているレコードを取得
                numx = tmp_df.query("code == 11")["num"].iloc[0]
                # 同義語の優先語と同義語を得る
                tmp_df = t_df.query("@numx == num").query("code in [10, 11]")
            # 同義語がなく優先語のみある場合
            elif 10 in {n: 0 for n in tmp_df["code"].to_list()}:
                numx = tmp_df.query("code == 10")["num"].iloc[0]
                tmp_df = t_df.query("@numx == num").query("code in [10, 11]")
            else:
                continue
            # 追加
            for w, t in zip(tmp_df["term"], tmp_df["token"]):
                dct["word"].append(w)
                dct["medis"].append(medis)
                # dct["icd"].append(icd)
                dct["source"].append("T")
                dct["token"].append(t)
    # to DataFrame
    aug_X_train = pd.DataFrame.from_dict(dct, orient='index').T
    # 訓練データと拡張データを結合
    X_train = pd.concat([X_train, aug_X_train])
    # 重複を削除
    X_train = X_train[~X_train.duplicated()]
    return X_train


def load_data(args, train_path="./data/train/train.csv", t_path="./data/train/t_df.csv"):
    # [UNK]だけになると何も学習できないので消します
    df = pd.read_csv(train_path)
    df = df[df["token"] != "[UNK]"]

    if args.icd:
        # 「病名表記」の情報だけでICDコードを選ぶことができない場合は、空欄となっています。https://www2.medis.or.jp/stdcd/byomei/spc509.pdf
        # 欠損を消すこと
        df = df.drop("medis", axis=1)
        df = df.rename(columns={'icd': 'medis'})
        # ICDが欠損してる場合は行を削除 
        df = df.dropna(subset=['medis'])

    t_df = pd.read_csv(t_path)
    t_df = t_df[t_df["token"] != "[UNK]"]
    return df, t_df


def cgi_medis2icd(cgi_df, medis_path="./data/medis/nmain504.csv"):
    # MEDISの辞書
    ma_df = pd.read_csv(medis_path)
    #
    medis2icd = {x: y for x, y in zip(ma_df["病名交換用コード"], ma_df["ICD10-2013"])}
    medis2icd["----"] = "----"
    #
    cgi_df["medis"] = [medis2icd[x] if x in medis2icd else "NO_MATCH" for x in cgi_df["medis"]]
    cgi_df = cgi_df[cgi_df["medis"] != "NO_MATCH"]

    return cgi_df