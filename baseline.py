from simstring.feature_extractor.character_ngram import CharacterNgramFeatureExtractor
from simstring.feature_extractor.word_ngram import WordNgramFeatureExtractor
from simstring.measure.cosine import CosineMeasure
from simstring.database.dict import DictDatabase
from simstring.searcher import Searcher
import os
import sys
import random
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
import json
import  argparse
import json, MeCab
import jaconv, unicodedata, regex, argparse
from tqdm import tqdm
from lib.util import load_data, data_augmentation, ExperimentDataset, my_collate_fn, cgi_medis2icd


def main():
    for fold in range(0, 5, 1):
        # データを読み込み
        X_train = pd.read_csv("{0}/X_train_{1}.csv".format(args.data_path, fold))
        X_test = pd.read_csv("{0}/X_test_{1}.csv".format(args.data_path, fold))
        # 内挿
        searcher, word2medis = interpotation_evaluation(X_train, X_test, fold, n_gram=2)
        # 外挿
        extrapolation_evaluration(searcher, X_train, word2medis, fold)


def interpotation_evaluation(X_train, X_test, fold, n_gram=2):
    # Simstringを構築
    db = DictDatabase(CharacterNgramFeatureExtractor(n_gram))
    for word in X_train["word"]:
        db.add(word)
    searcher = Searcher(db, CosineMeasure())
    word2medis = {x: y for x, y in zip(X_train["word"], X_train["medis"])}
    # テストデータで評価
    labels, preds = [], []
    for word, medis in zip(X_test["word"], X_test["medis"]):
        labels.append(medis)
        results = searcher.search(word, args.threshold)
        if len(results) > 0:
            preds.append(word2medis[results[0]])
        else:
            preds.append("NONE")
    # 評価して保存
    res_dct = classification_report(labels, preds, output_dict=True)
    # 保存
    if args.icd:
        with open('./test_results/{1}/test_results_icd_{0}.json'.format(fold, args.exp_name), 'w') as f:
            json.dump(res_dct, f, indent=4)
    else:
        with open('./test_results/{1}/test_results_medis_{0}.json'.format(fold, args.exp_name), 'w') as f:
            json.dump(res_dct, f, indent=4)
    return searcher, word2medis


def extrapolation_evaluration(searcher, X_train, word2medis, fold):
    # CGIを読み込む
    cgi_df = pd.read_csv("./data/cgi/cgi_raw_data.csv")
    cgi_df= cgi_df.drop_duplicates(subset=["input", "gold", "medis"])
    # UNKを弾く
    cgi_df = cgi_df[cgi_df["gold"] != "unknown"]
    if args.icd:
        cgi_df = cgi_medis2icd(cgi_df)
    # 学習にないものは除外
    codes = list(X_train["medis"].unique())
    cgi_df = cgi_df.query('medis in @codes')
    # CGIで評価
    labels, preds = [], []
    for word, medis in zip(cgi_df["input"], cgi_df["medis"]):
        labels.append(medis)
        results = searcher.search(word, args.threshold)
        if len(results) > 0:
            preds.append(word2medis[results[0]])
        else:
            preds.append("NONE")
    # 評価して保存
    res_dct = classification_report(labels, preds, output_dict=True)
    # 保存
    if args.icd:
        with open('./cgi_results/{1}/cgi_results_icd_{0}.json'.format(fold, args.exp_name), 'w') as f:
            json.dump(res_dct, f, indent=4)
    else:
        with open('./cgi_results/{1}/cgi_results_medis_{0}.json'.format(fold, args.exp_name), 'w') as f:
            json.dump(res_dct, f, indent=4)


if __name__ == '__main__':
    # 引数
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default="simstring")
    parser.add_argument('--data_path', type=str, default="./results/bert_cls_medis")
    parser.add_argument('--threshold', type=float, default=0.7)
    parser.add_argument('--icd', action='store_true')
    args = parser.parse_args()
    # フォルダ作成
    for SAMPLE_DIR in ["./cgi_results/{0}".format(args.exp_name), "./test_results/{0}".format(args.exp_name)]:
        if not os.path.exists(SAMPLE_DIR):
            os.makedirs(SAMPLE_DIR)
    main()
